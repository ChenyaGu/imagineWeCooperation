import os
import pickle
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# dirName = os.getcwd()
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# import xmltodict
# import mujoco_py as mujoco
import pandas as pd
import itertools as it
from collections import OrderedDict
import numpy as np
import glob
from src.writer import loadFromPickle,saveToPickle


# print(data)
# print(data[0])

from src.mathTools.distribution import sampleFromDistribution,  SoftDistribution, BuildGaussianFixCov, sampleFromContinuousSpace
from src.inference.intention import UpdateIntention
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsContinuousPolicyLikelihood, InferOneStep
from src.inference.percept import SampleNoisyAction, PerceptImaginedWeAction
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpVariousForce, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe, ReshapeActionVariousForce ,ResetMultiAgentNewtonChasingVariousSheep,ReshapeHumanAction
from collections import OrderedDict
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoNoisy
from src.functionTools.loadSaveModel import  restoreVariables, GetSavePath
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.MDPChasing.policy import RandomPolicy
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, GetActionFromJointActionDistribution, HierarchyPolicyForCommittedAgent, SampleIndividualActionGivenIntention, GetIntensionOnChangableIntention, SampleActionOnFixedIntention, SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes

class ComposeCentralControlPolicyByGaussianOnDeterministicAction:
    def __init__(self, reshapeAction, observe, actOneStepOneModel, buildGaussian):
        self.reshapeAction = reshapeAction
        self.observe = observe
        self.actOneStepOneModel = actOneStepOneModel
        self.buildGaussian = buildGaussian

    def __call__(self, individualModels, numAgentsInWe):
        centralControlPolicy = lambda state: [self.buildGaussian(tuple(self.reshapeAction(
            self.actOneStepOneModel(individualModels[agentId], self.observe(state))))) for agentId in range(numAgentsInWe)] 
        return centralControlPolicy



def calCulateIntenTion(trajDictList,numSheep,fileName):
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [1]
    manipulatedVariables['valuePriorSoftMaxBeta'] = [0.0]
    manipulatedVariables['valuePriorEndTime'] = [-100]
    manipulatedVariables['deviationFor2DAction'] = [1.0]#, 3.0, 9.0]
    manipulatedVariables['rationalityBetaInInference'] = [1.0]#[0.0, 0.1, 0.2, 0.5, 1.0]
    parameters = manipulatedVariables


    deviationFor2DAction = parameters['deviationFor2DAction'][0]
    rationalityBetaInInference = parameters['rationalityBetaInInference'][0]
    valuePriorEndTime = parameters['valuePriorEndTime'][0]
    numWolves = 3
    numBlocks = 0
    numAgents = numWolves + numSheep
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))
    possibleWolvesIds = wolvesID
    possibleSheepIds = sheepsID


    mapSize = 1
    minDistance = mapSize * 1 / 3 
    reset = ResetMultiAgentNewtonChasingVariousSheep(numWolves, numBlocks, mapSize, minDistance)

    intentionSpacesForAllWolves = [tuple(it.product(possibleSheepIds, [tuple(possibleWolvesIds)])) 
                    for wolfId in possibleWolvesIds]
    # print(intentionSpacesForAllWolves)
    wolvesIntentionPriors = [{tuple(intention): 1/len(allPossibleIntentionsOneWolf) for intention in allPossibleIntentionsOneWolf} 
            for allPossibleIntentionsOneWolf in intentionSpacesForAllWolves] 
    # Percept Action For Inference
    #perceptAction = lambda action: action
    perceptSelfAction = SampleNoisyAction(deviationFor2DAction)
    perceptOtherAction = SampleNoisyAction(deviationFor2DAction)
    perceptAction = PerceptImaginedWeAction(possibleWolvesIds, perceptSelfAction, perceptOtherAction)
    #perceptAction = lambda action: action


    # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
    # ------------ model ------------------------
    weModelsListBaseOnNumInWe = []
    observeListBaseOnNumInWe = []
    for numAgentInWe in range(numWolves, numWolves + 1):
            worldDim = 2
            actionDim = worldDim * 2 + 1
            numBlocksForWe = 0
            wolvesIDForWolfObserve = list(range(numAgentInWe))
            sheepsIDForWolfObserve = list(range(numAgentInWe, 1 + numAgentInWe))
            blocksIDForWolfObserve = list(range(1 + numAgentInWe, 1 + numAgentInWe + numBlocksForWe))
            observeOneAgentForWolf = lambda agentID: Observe(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve, 
                    blocksIDForWolfObserve, getPosFromAgentState, getVelFromAgentState)
            observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in range(numAgentInWe + 1)]
            observeListBaseOnNumInWe.append(observeWolf)

            obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
            initObsForWolfParams = observeWolf(reset(numSheep)[obsIDsForWolf])
            obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
            buildWolfModels = BuildMADDPGModels(actionDim, numAgentInWe + 1, obsShapeWolf)
            layerWidthForWolf = [64 * (numAgentInWe - 1), 64 * (numAgentInWe - 1)]
            wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numAgentInWe)]
            
            # if wolfType == 'sharedAgencyByIndividualRewardWolf':
            #         wolfPrefix = 'maddpgIndividWolf'
            # if wolfType == 'sharedAgencyBySharedRewardWolf':
            #         wolfPrefix = 'maddpg'
            # wolfFileName = "{}wolves{}sheep{}blocks{}eps_agent".format(numAgentInWe, 1, numBlocksForWe, maxEpisode)
            modelFolderName = 'withoutWall3wolvesForModel'
            modelFolder = os.path.join(dirName, '..','..', 'model', modelFolderName)
            numBlocks = 0
            maxEpisode = 60000
            maxTimeStep = 75
            modelSheepSpeed = 1.0
            # wolfModelPaths = [os.path.join(dirName, '..', '..', 'data', 'preTrainModel', wolfPrefix + wolfFileName + str(i) + '60000eps') for i in range(numAgentInWe)]
            wolfFileName = "maddpg{}wolves1sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
            evaluateEpisode = 60000
            # wolfModelPaths = [os.path.join(modelFolder, 'trainingId' + str(i) + wolfFileName + str(evaluateEpisode) + 'eps') for i in range(numWolves)]
            wolfModelPaths = [os.path.join(modelFolder, wolfFileName + str(i) + str(evaluateEpisode) + 'eps') for i in range(numWolves)]
            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
            weModelsListBaseOnNumInWe.append(wolfModelsList)
            print('loadModel',len(weModelsListBaseOnNumInWe))

    actionDimReshaped = 2
    cov = [deviationFor2DAction ** 2 for _ in range(actionDimReshaped)]
    buildGaussian = BuildGaussianFixCov(cov)
    actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoNoisy)
    #actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoisy)
    reshapeAction = ReshapeHumanAction()
    composeCentralControlPolicy = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction, 
            observe, actOneStepOneModelWolf, buildGaussian) 
    wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 3])(weModelsListBaseOnNumInWe[numAgentsInWe - 3], numAgentsInWe) 
            for numAgentsInWe in range(numWolves, numWolves + 1)]
    # wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 2])(weModelsListBaseOnNumInWe[numAgentsInWe - 2], numAgentsInWe) 
            # for numAgentsInWe in range(2, numWolves + 1)]
    centralControlPolicyListBasedOnNumAgentsInWe = wolvesCentralControlPolicies # 0 for two agents in We, 1 for three agents...
    softPolicyInInference = lambda distribution : distribution
    getStateThirdPersonPerspective = lambda state, goalId, weIds: getStateOrActionThirdPersonPerspective(state, goalId, weIds, blocksID)
    policyForCommittedAgentsInInference = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe, softPolicyInInference,
            getStateThirdPersonPerspective)
    concernedAgentsIds = possibleWolvesIds
    calCommittedAgentsPolicyLikelihood = CalCommittedAgentsContinuousPolicyLikelihood(concernedAgentsIds, 
            policyForCommittedAgentsInInference, rationalityBetaInInference)

    randomActionSpace = [(5, 0), (3.5, 3.5), (0, 5), (-3.5, 3.5), (-5, 0), (-3.5, -3.5), (0, -5), (3.5, -3.5), (0, 0)]
    randomPolicy = RandomPolicy(randomActionSpace)
    getStateFirstPersonPerspective = lambda state, goalId, weIds, selfId: getStateOrActionFirstPersonPerspective(state, goalId, weIds, selfId, blocksID)
    policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy, 
            softPolicyInInference, getStateFirstPersonPerspective)
    calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(possibleWolvesIds, 
            concernedAgentsIds, policyForUncommittedAgentsInInference)

    # Joint Likelihood
    calJointLikelihood = lambda intention, state, perceivedAction: calCommittedAgentsPolicyLikelihood(intention, state, perceivedAction) * \
            calUncommittedAgentsPolicyLikelihood(intention, state, perceivedAction)

    # Infer and update Intention
    variablesForAllWolves = [[intentionSpace] for intentionSpace in intentionSpacesForAllWolves]
    jointHypothesisSpaces = [pd.MultiIndex.from_product(variables, names=['intention']) for variables in variablesForAllWolves]
    concernedHypothesisVariable = ['intention']
    # priorDecayRate = 1  # record all prior
    priorDecayRate = 0.7
    softPrior = SoftDistribution(priorDecayRate)
    inferIntentionOneStepList = [InferOneStep(jointHypothesisSpace, concernedHypothesisVariable, 
            calJointLikelihood, softPrior) for jointHypothesisSpace in jointHypothesisSpaces]

    if numSheep == 1:
            inferIntentionOneStepList = [lambda prior, state, action: prior] * 3

    adjustIntentionPriorGivenValueOfState = lambda state: 1
    chooseIntention = sampleFromDistribution
    updateIntentions = [UpdateIntention(intentionPrior, valuePriorEndTime, adjustIntentionPriorGivenValueOfState, 
            perceptAction, inferIntentionOneStep, chooseIntention) 
            for intentionPrior, inferIntentionOneStep in zip(wolvesIntentionPriors, inferIntentionOneStepList)]

    # reset intention and adjuste intention prior attributes tools for multiple trajectory
    intentionResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
    intentionResetAttributeValues = [dict(zip(intentionResetAttributes, [0, None, None, intentionPrior, [intentionPrior]]))
            for intentionPrior in wolvesIntentionPriors]
    resetIntentions = ResetObjects(intentionResetAttributeValues, updateIntentions)
    returnAttributes = ['formerIntentionPriors']
    getIntentionDistributions = GetObjectsValuesOfAttributes(returnAttributes, updateIntentions)
    attributesToRecord = ['lastAction']
    recordActionForUpdateIntention = RecordValuesForObjects(attributesToRecord, updateIntentions) 

    # Wolves Generate Action
    covForPlanning = [0.03 ** 2 for _ in range(actionDimReshaped)]
    buildGaussianForPlanning = BuildGaussianFixCov(covForPlanning)
    composeCentralControlPolicyForPlanning = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction,observe, actOneStepOneModelWolf, buildGaussianForPlanning) 
    wolvesCentralControlPoliciesForPlanning = [composeCentralControlPolicyForPlanning(
            observeListBaseOnNumInWe[numAgentsInWe - 3])(weModelsListBaseOnNumInWe[numAgentsInWe - 3], numAgentsInWe) 
            for numAgentsInWe in range(numWolves, numWolves + 1)]

    centralControlPolicyListBasedOnNumAgentsInWeForPlanning = wolvesCentralControlPoliciesForPlanning # 0 for two agents in We, 1 for three agents...
    softPolicyInPlanning = lambda distribution: distribution
    policyForCommittedAgentInPlanning = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWeForPlanning, softPolicyInPlanning,
            getStateThirdPersonPerspective)

    policyForUncommittedAgentInPlanning = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy, softPolicyInPlanning,
            getStateFirstPersonPerspective)

    def wolfChooseActionMethod(individualContinuousDistributions):
            centralControlAction = tuple([tuple(sampleFromContinuousSpace(distribution)) 
            for distribution in individualContinuousDistributions])
            return centralControlAction

    getSelfActionThirdPersonPerspective = lambda weIds, selfId : list(weIds).index(selfId)
    chooseCommittedAction = GetActionFromJointActionDistribution(wolfChooseActionMethod, getSelfActionThirdPersonPerspective)
    chooseUncommittedAction = sampleFromDistribution
    wolvesSampleIndividualActionGivenIntentionList = [SampleIndividualActionGivenIntention(selfId, policyForCommittedAgentInPlanning,policyForUncommittedAgentInPlanning, chooseCommittedAction, chooseUncommittedAction) 
            for selfId in possibleWolvesIds]

    # Sample and Save Trajectory
    # trajectoriesWithIntentionDists = []
    # trajectory = data[0]['trajectory']
    # print( data[1]['condition'],len(data[1]['trajectory']))
    wolvesSampleActions = [GetIntensionOnChangableIntention(updateIntention, wolvesSampleIndividualActionGivenIntention) 
                        for updateIntention, wolvesSampleIndividualActionGivenIntention in zip(updateIntentions, wolvesSampleIndividualActionGivenIntentionList)]
    reshapeHumanAction = ReshapeHumanAction()
    trajectoriesWithIntentionDists = []

    for trajDict in trajDictList:
        trajectoryDictWithIntension = trajDict.copy()
        if trajDict['condition']['sheepNums'] == numSheep:
            trajectory = trajDict['trajectory']
            for state in trajectory:
                    action = state[1]
                    # print(action)
                    intention = [sampleAction(state[0]) for sampleAction in wolvesSampleActions]
                    reshapedAction =[reshapeHumanAction(sigAction) for sigAction in action]
                    recordActionForUpdateIntention([reshapedAction]) 
                    # intentionDistributions = getIntentionDistributions()
                    # print(intentionDistributions,intention)
            intentionDistributions = getIntentionDistributions()
            trajectoryWithIntentionDists = [tuple(list(SASRPair) + list(intentionDist)) 
                    for SASRPair, intentionDist in zip(trajectory, intentionDistributions)]
            trajectoryDictWithIntension['trajectory'] = trajectoryWithIntentionDists
            trajectoryDictWithIntension['Name'] = fileName
            trajectoriesWithIntentionDists.append(trajectoryDictWithIntension) 
            # print(trajectoryWithIntentionDists)
            resetIntentions()
    print('finish:{}, numSheep:{}'.format(fileName,numSheep))
    return trajectoriesWithIntentionDists
# print(data[1]['condition'])
# print(trajectoryWithIntentionDists[0][3],trajectoryWithIntentionDists[-1][3])
# trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps}


if __name__ == "__main__":
    dataPath = os.path.join(dirName,'..','..','results')

    # loadDataPath = os.path.join(dataPath,'rawResults')
    # saveDataPath =  os.path.join(dataPath,'resultsWithIntension')

    # loadDataPath = os.path.join(dataPath,'rawMachineResults')
    # saveDataPath = os.path.join(dataPath,'machineResultsWithIntention')

    loadDataPath = os.path.join(dataPath,'testFolderIn')
    saveDataPath = os.path.join(dataPath,'testFolderOut')

    # resultPath = glob.glob(os.path.join(loadDataPath, '*.pickle'))
    fileNameList = os.listdir(loadDataPath)
    # print(a)
    for fileName in fileNameList:
        print(fileName)
        pickleWithIntention = []
        data = loadFromPickle(os.path.join(loadDataPath,fileName))  
        for numSheep in [1,2,4]:
            trajWithIntentionList = calCulateIntenTion(data,numSheep,fileName)
            pickleWithIntention = pickleWithIntention + trajWithIntentionList

        print(pickleWithIntention[0]['trajectory'][-1])
        saveToPickle(pickleWithIntention, os.path.join(saveDataPath,fileName))
        print('save:',os.path.join(saveDataPath,fileName))
    # data = loadFromPickle(resultPath[0])  