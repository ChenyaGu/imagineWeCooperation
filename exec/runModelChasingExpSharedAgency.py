import os
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import collections as co
import itertools as it
import functools as ft
from collections import OrderedDict
import pandas as pd

import numpy as np
import random
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback, InitializeScreen, \
    DrawAttributionTrail, DrawImageWithJoysticksCheck, DrawNewStateWithBlocks
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrialAllCondtionVariouSpeedForModel, isAnyKilled, CheckTerminationOfTrial, RecordEatenNumber
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy, actByPolicyTrainNoNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
from src.mathTools.distribution import sampleFromDistribution,  SoftDistribution, BuildGaussianFixCov, sampleFromContinuousSpace
# from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpWithNoise, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe, ReshapeHumanAction, ReshapeActionVariousForce, ResetMultiAgentNewtonChasingVariousSheep, \
    BuildGaussianFixCov, sampleFromContinuousSpace, ComposeCentralControlPolicyByGaussianOnDeterministicAction
from src.MDPChasing.policy import RandomPolicy
from src.inference.intention import UpdateIntention
from src.inference.percept import SampleNoisyAction, PerceptImaginedWeAction
from src.inference.inference import CalUncommittedAgentsPolicyLikelihood, CalCommittedAgentsContinuousPolicyLikelihood, InferOneStep
from src.MDPChasing.state import getStateOrActionFirstPersonPerspective, getStateOrActionThirdPersonPerspective
from src.generateAction.imaginedWeSampleAction import PolicyForUncommittedAgent, PolicyForCommittedAgent, GetActionFromJointActionDistribution, HierarchyPolicyForCommittedAgent, SampleIndividualActionGivenIntention, GetIntensionOnChangableIntention, SampleActionOnFixedIntention, SampleActionMultiagent
from src.sampleTrajectoryTools.resetObjectsForMultipleTrjaectory import RecordValuesForObjects, ResetObjects, GetObjectsValuesOfAttributes



def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [1, 2, 4]
    manipulatedVariables['sheepWolfForceRatio'] = [1.2]
    manipulatedVariables['sheepConcern'] = ['self']
    # manipulatedVariables['sheepConcern'] = ['self', 'all']
    trailNumEachCondition = 20
    deviationFor2DAction = 1.0
    rationalityBetaInInference = 1.0
    valuePriorEndTime = -100

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    # random.shuffle(AllConditions)

    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter players' name:").capitalize()

    mapSize = 1.0
    displaySize = 1.0
    minDistance = mapSize * 1 / 3
    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2

    screenWidth = int(800)
    screenHeight = int(800)
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    targetColor = [THECOLORS['orange'], THECOLORS['chocolate1'], THECOLORS['tan1'], THECOLORS['goldenrod2']]
    #'orange', (255, 165, 0); 'chocolate1', (255, 127, 36); 'tan1', (255, 165, 79); 'goldenrod1', (255, 193, 37)
    # targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['red3'], THECOLORS['blue3'], THECOLORS['green4']]
    blockColors = [THECOLORS['white']] * 2
    textColorTuple = THECOLORS['green']

    gridSize = 40
    leaveEdgeSpace = 5
    playerRadius = int(wolfSize/(displaySize*2)*screenWidth*gridSize/(gridSize+2*leaveEdgeSpace))
    targetRadius = int(sheepSize/(displaySize*2)*screenWidth*gridSize/(gridSize+2*leaveEdgeSpace))
    blockRadius = int(blockSize/(displaySize*2)*screenWidth*gridSize/(gridSize+2*leaveEdgeSpace))
    stopwatchUnit = 100
    finishTime = 1000 * 18
    stopwatchEvent = pg.USEREVENT + 1

    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    pg.key.set_repeat(120, 120)
    picturePath = os.path.abspath(os.path.join(os.path.join(dirName, '..'), 'pictures'))
    resultsDicPath = os.path.join(dirName, '..', 'results')
    
    # experimentValues["name"] = '0704'
    writerPath = os.path.join(resultsDicPath, experimentValues["name"]) + '.csv'
    picklePath = os.path.join(resultsDicPath, experimentValues["name"]) + '.pickle'
    writer = WriteDataFrameToCSV(writerPath)
    pickleWriter = lambda data: saveToPickle(data,picklePath)
    # introductionImage = pg.image.load(os.path.join(picturePath, 'introduction-waitall.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest-waitall.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    # introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))

    # --------environment setting-----------
    numWolves = 3
    experimentValues["numWolves"] = numWolves
    numBlocks = 0
    reset = ResetMultiAgentNewtonChasingVariousSheep(numWolves, numBlocks, mapSize, minDistance)

    allSheepPolicy = {}
    allWolfPolicy = {}
    for numSheeps in manipulatedVariables['sheepNums']:
        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numAgents))
        blocksID = list(range(numAgents, numEntities))
        possibleWolvesIds = wolvesID
        possibleSheepIds = sheepsID

        intentionSpacesForAllWolves = [tuple(it.product(possibleSheepIds, [tuple(possibleWolvesIds)]))
                                       for wolfId in possibleWolvesIds]
        # print(intentionSpacesForAllWolves)
        wolvesIntentionPriors = [{tuple(intention): 1 / len(allPossibleIntentionsOneWolf) for intention in allPossibleIntentionsOneWolf}
            for allPossibleIntentionsOneWolf in intentionSpacesForAllWolves]
        # Percept Action For Inference
        # perceptAction = lambda action: action
        perceptSelfAction = SampleNoisyAction(deviationFor2DAction)
        perceptOtherAction = SampleNoisyAction(deviationFor2DAction)
        perceptAction = PerceptImaginedWeAction(possibleWolvesIds, perceptSelfAction, perceptOtherAction)

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        # ------------ model ------------------------
        weModelsListBaseOnNumInWe = []
        observeListBaseOnNumInWe = []
        for numAgentInWe in range(numWolves, numWolves + 1):
            worldDim = 2
            actionDim = worldDim * 2 + 1
            numSheepForWe = 1
            numBlocksForWe = 0
            wolvesIDForWolfObserve = list(range(numAgentInWe))
            sheepsIDForWolfObserve = list(range(numAgentInWe, 1 + numAgentInWe))
            blocksIDForWolfObserve = list(range(1 + numAgentInWe, 1 + numAgentInWe + numBlocksForWe))
            observeOneAgentForWolf = lambda agentID: Observe(agentID, wolvesIDForWolfObserve, sheepsIDForWolfObserve,
                                                             blocksIDForWolfObserve, getPosFromAgentState,
                                                             getVelFromAgentState)
            observeWolf = lambda state: [observeOneAgentForWolf(agentID)(state) for agentID in range(numAgentInWe + 1)]
            observeListBaseOnNumInWe.append(observeWolf)

            obsIDsForWolf = wolvesIDForWolfObserve + sheepsIDForWolfObserve + blocksIDForWolfObserve
            initObsForWolfParams = observeWolf(reset(numSheepForWe)[obsIDsForWolf])
            obsShapeWolf = [initObsForWolfParams[obsID].shape[0] for obsID in range(len(initObsForWolfParams))]
            buildWolfModels = BuildMADDPGModels(actionDim, numAgentInWe + 1, obsShapeWolf)
            layerWidthForWolf = [64 * (numAgentInWe - 1), 64 * (numAgentInWe - 1)]
            wolfModelsList = [buildWolfModels(layerWidthForWolf, agentID) for agentID in range(numAgentInWe)]

            modelFolderName = 'withoutWall3wolvesForModel'
            modelFolder = os.path.join(dirName, '..', '..', 'model', modelFolderName)
            numBlocks = 0
            maxEpisode = 60000
            maxTimeStep = 75
            modelSheepSpeed = 1.0
            wolfFileName = "maddpg{}wolves1sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves,
                                                                                                         numBlocks,
                                                                                                         maxEpisode,
                                                                                                         maxTimeStep,
                                                                                                         modelSheepSpeed)
            evaluateEpisode = 60000
            wolfModelPaths = [os.path.join(modelFolder, wolfFileName + str(i) + str(evaluateEpisode) + 'eps') for i in
                              range(numWolves)]
            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
            weModelsListBaseOnNumInWe.append(wolfModelsList)
            print('loadModel', len(weModelsListBaseOnNumInWe))

        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

        wolfMaxSpeed = 1
        sheepMaxSpeed = 1.3
        blockMaxSpeed = None

        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities

        # reshapeHumanAction = ReshapeHumanAction()
        # reshapeSheepAction = ReshapeSheepAction()
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                              getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                        getVelFromAgentState, getPosFromAgentState)

        actionDimReshaped = 2
        cov = [0.3 ** 2 for _ in range(actionDimReshaped)]
        buildGaussian = BuildGaussianFixCov(cov)
        actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoNoisy)
        # actOneStepOneModelWolf = ActOneStep(actByPolicyTrainNoisy)
        reshapeAction = ReshapeHumanAction()
        composeCentralControlPolicy = lambda observe: ComposeCentralControlPolicyByGaussianOnDeterministicAction(reshapeAction,
            observe, actOneStepOneModelWolf, buildGaussian)
        wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 3])(
            weModelsListBaseOnNumInWe[numAgentsInWe - 3], numAgentsInWe)
                                        for numAgentsInWe in range(numWolves, numWolves + 1)]
        # wolvesCentralControlPolicies = [composeCentralControlPolicy(observeListBaseOnNumInWe[numAgentsInWe - 2])(weModelsListBaseOnNumInWe[numAgentsInWe - 2], numAgentsInWe)
        # for numAgentsInWe in range(2, numWolves + 1)]
        centralControlPolicyListBasedOnNumAgentsInWe = wolvesCentralControlPolicies  # 0 for two agents in We, 1 for three agents...
        softPolicyInInference = lambda distribution: distribution
        getStateThirdPersonPerspective = lambda state, goalId, weIds: getStateOrActionThirdPersonPerspective(state,
                                                                                                             goalId,
                                                                                                             weIds,
                                                                                                             blocksID)
        policyForCommittedAgentsInInference = PolicyForCommittedAgent(centralControlPolicyListBasedOnNumAgentsInWe,
                                                                      softPolicyInInference,
                                                                      getStateThirdPersonPerspective)
        concernedAgentsIds = possibleWolvesIds
        calCommittedAgentsPolicyLikelihood = CalCommittedAgentsContinuousPolicyLikelihood(concernedAgentsIds,
                                                                                          policyForCommittedAgentsInInference,
                                                                                          rationalityBetaInInference)

        randomActionSpace = [(5, 0), (3.5, 3.5), (0, 5), (-3.5, 3.5), (-5, 0), (-3.5, -3.5), (0, -5), (3.5, -3.5), (0, 0)]
        randomPolicy = RandomPolicy(randomActionSpace)
        getStateFirstPersonPerspective = lambda state, goalId, weIds, selfId: getStateOrActionFirstPersonPerspective(
            state, goalId, weIds, selfId, blocksID)
        policyForUncommittedAgentsInInference = PolicyForUncommittedAgent(possibleWolvesIds, randomPolicy,
                                                                          softPolicyInInference,
                                                                          getStateFirstPersonPerspective)
        calUncommittedAgentsPolicyLikelihood = CalUncommittedAgentsPolicyLikelihood(possibleWolvesIds,
                                                                                    concernedAgentsIds,
                                                                                    policyForUncommittedAgentsInInference)
        # Joint Likelihood
        calJointLikelihood = lambda intention, state, perceivedAction: calCommittedAgentsPolicyLikelihood(intention, state, perceivedAction) * \
                                                                       calUncommittedAgentsPolicyLikelihood(intention, state, perceivedAction)

        # Infer and update Intention
        variablesForAllWolves = [[intentionSpace] for intentionSpace in intentionSpacesForAllWolves]
        jointHypothesisSpaces = [pd.MultiIndex.from_product(variables, names=['intention']) for variables in
                                 variablesForAllWolves]
        concernedHypothesisVariable = ['intention']
        priorDecayRate = 1
        softPrior = SoftDistribution(priorDecayRate)
        inferIntentionOneStepList = [InferOneStep(jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood, softPrior) for jointHypothesisSpace in
                                     jointHypothesisSpaces]

        if numSheeps == 1:
            inferIntentionOneStepList = [lambda prior, state, action: prior] * 3

        adjustIntentionPriorGivenValueOfState = lambda state: 1
        chooseIntention = sampleFromDistribution
        updateIntentions = [UpdateIntention(intentionPrior, valuePriorEndTime, adjustIntentionPriorGivenValueOfState,
                                            perceptAction, inferIntentionOneStep, chooseIntention)
                            for intentionPrior, inferIntentionOneStep in
                            zip(wolvesIntentionPriors, inferIntentionOneStepList)]

        # reset intention and adjuste intention prior attributes tools for multiple trajectory
        intentionResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
        intentionResetAttributeValues = [
            dict(zip(intentionResetAttributes, [0, None, None, intentionPrior, [intentionPrior]]))
            for intentionPrior in wolvesIntentionPriors]
        resetIntentions = ResetObjects(intentionResetAttributeValues, updateIntentions)

        def loadPolicyOneCondition(numSheeps, sheepConcern):
            # -----------observe--------
            if sheepConcern == 'self':
                numSheepToObserve = 1
            if sheepConcern == 'all':
                numSheepToObserve = numSheeps

            observeOneAgent1 = lambda agentID, sId: Observe(agentID, wolvesID, sId, blocksID, getPosFromAgentState,
                                                            getVelFromAgentState)
            observeOneAgent = ft.partial(observeOneAgent1, sId=sheepsID)
            observeOne = lambda state, num: [observeOneAgent(agentID)(state) for agentID in range(num)]
            observe = ft.partial(observeOne, num=numAgents)
            initObsForParams = observe(reset(numSheeps))
            obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

            wolvesIDForSheepObserve = list(range(numWolves))
            sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
            blocksIDForSheepObserve = list(
                range(numSheepToObserve + numWolves, numSheepToObserve + numWolves + numBlocks))
            observeOneAgentForSheep1 = lambda agentID, sId: Observe(agentID, wolvesIDForSheepObserve, sId,
                                                                    blocksIDForSheepObserve,
                                                                    getPosFromAgentState, getVelFromAgentState)
            observeOneAgentForSheep = ft.partial(observeOneAgentForSheep1, sId=sheepsIDForSheepObserve)
            observeOneForSheep = lambda state, num: [observeOneAgentForSheep(agentID)(state) for agentID in
                                                     range(num)]
            sheepObserve = ft.partial(observeOneForSheep, num=numWolves + numSheepToObserve)
            sheepObsList = []
            for sheepId in sheepsID:
                obsFunList = [Observe(agentID, wolvesIDForSheepObserve, [sheepId], blocksIDForSheepObserve,
                                      getPosFromAgentState,
                                      getVelFromAgentState) for agentID in list(range(numWolves)) + [sheepId]]
                sheepObsLambda = lambda state, obsList: list([obs(state) for obs in obsList])
                sheepObs = ft.partial(sheepObsLambda, obsList=obsFunList)
                sheepObsList.append(sheepObs)
            initSheepObsForParams = sheepObserve(reset(numSheepToObserve))
            obsSheepShape = [initSheepObsForParams[obsID].shape[0] for obsID in range(len(initSheepObsForParams))]

            worldDim = 2
            actionDim = worldDim * 2 + 1
            layerWidth = [128, 128]

            # -----------model--------
            modelFolderName = 'withoutWall3wolvesForModel'
            # modelFolderName = 'withoutWall2wolves'

            maxEpisode = 60000
            evaluateEpisode = 60000
            maxTimeStep = 75
            modelSheepSpeed = 1.0

            buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
            wolfModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves)]

            buildSheepMADDPGModels = BuildMADDPGModels(actionDim, numWolves + numSheepToObserve, obsSheepShape)
            sheepModelsListAll = [buildSheepMADDPGModels(layerWidth, agentID) for agentID in
                                  range(numWolves, numWolves + numSheepToObserve)]
            sheepModelsListSep = [buildSheepMADDPGModels(layerWidth, agentID) for agentID in
                                  range(numWolves, numWolves + numSheepToObserve) for i in range(numSheeps)]

            modelFolder = os.path.join(dirName, '..', 'model', modelFolderName)
            wolfFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves,
                                                                                                          numSheeps,
                                                                                                          numBlocks,
                                                                                                          maxEpisode,
                                                                                                          maxTimeStep,
                                                                                                          modelSheepSpeed)
            sheepFileNameSep = "maddpg{}wolves1sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent3".format(numWolves,
                                                                                                              numBlocks,
                                                                                                              maxEpisode,
                                                                                                              maxTimeStep,
                                                                                                              modelSheepSpeed)
            sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0individ1.0_agent".format(
                numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)

            wolfModelPaths = [os.path.join(modelFolder, wolfFileName + str(i) + str(evaluateEpisode) + 'eps') for i in
                              range(numWolves)]
            sheepModelPathsAll = [os.path.join(modelFolder, sheepFileNameAll + str(i) + str(evaluateEpisode) + 'eps')
                                  for i in range(numWolves, numWolves + numSheepToObserve)]
            sheepModelPathsSep = [
                os.path.join(modelFolder, 'trainingId' + str(i) + sheepFileNameSep + str(evaluateEpisode) + 'eps') for i
                in range(numSheeps)]

            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)

            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
            wolfPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in
                                                          wolfModelsList]
            wolfPolicyOneCondition = ft.partial(wolfPolicyFun, obs=observe)

            if numSheepToObserve == 1:
                [restoreVariables(model, path) for model, path in zip(sheepModelsListSep, sheepModelPathsSep)]
                sheepPolicyFun = lambda allAgentsStates: list(
                    [actOneStepOneModel(model, sheepObsList[i](allAgentsStates)) for i, model in
                     enumerate(sheepModelsListSep)])
                sheepPolicyOneCondition = sheepPolicyFun
            else:
                [restoreVariables(model, path) for model, path in zip(sheepModelsListAll, sheepModelPathsAll)]
                sheepPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model
                                                               in sheepModelsListAll]
                sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=sheepObserve)
            return sheepPolicyOneCondition, wolfPolicyOneCondition
        for sheepConcern in manipulatedVariables['sheepConcern']:
            sheepPolicy, wolfPolicy = loadPolicyOneCondition(numSheeps, sheepConcern)
            allSheepPolicy.update({(numSheeps, sheepConcern): sheepPolicy})
            allWolfPolicy.update({(numSheeps, sheepConcern): wolfPolicy})


        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-displaySize, displaySize], [-displaySize, displaySize])
        def checkBoudary(agentState):
            newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
            return newState
        checkAllAgents = lambda states: [checkBoudary(agentState) for agentState in states]
        reShapeAction = ReshapeActionVariousForce()
        noiseAction = lambda state: sampleFromContinuousSpace(buildGaussian(tuple(state)))
        transit = TransitMultiAgentChasingForExpWithNoise(reShapeAction, reShapeAction, applyActionForce,
                                                          applyEnvironForce, integrateState, checkAllAgents,
                                                          noiseAction)
        # transit = TransitMultiAgentChasingForExpVariousForce(reShapeAction, reShapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone = wolfSize + sheepSize
    recordEaten = RecordEatenNumber(isAnyKilled)
    modelController = allWolfPolicy
    # humanController = JoyStickForceControllers()
    # drawImageBoth = DrawImageWithJoysticksCheck(screen,humanController.joystickList)
    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
    drawNewState = DrawNewStateWithBlocks(screen, drawBackground, playerColors, blockColors, targetRadius, playerRadius, blockRadius, displaySize)
    drawImage = DrawImage(screen)
    trial = NewtonChaseTrialAllCondtionVariouSpeedForModel(screen, killzone, targetColor, numWolves, numBlocks, stopwatchEvent,
                                                           drawNewState, checkTerminationOfTrial, recordEaten, modelController,
                                                           getEntityPos, getEntityVel, allSheepPolicy, transit)

    hasRest = False  # True
    experiment = NewtonExperiment(restImage, hasRest, trial, writer, pickleWriter, experimentValues, reset, drawImage)
    # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    # drawImageBoth(introductionImage)
    block = 1
    restTimes = 3  # the number of breaks in an experiment
    for i in range(block):
        experiment(finishTime, AllConditions, restTimes)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        else:
            drawImage(restImage)


if __name__ == "__main__":
    main()