import os
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import collections as co
import itertools as it
import functools as ft

import numpy as np
import random
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackgroundWithStep, DrawNewStateWithBlocksAndFeedback, DrawImage, GiveExperimentFeedback, \
    InitializeScreen, DrawAttributionTrail, DrawImageWithJoysticksCheck
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrialAllCondtionVariouSpeedForModel, isAnyKilled, CheckTerminationOfTrial, RecordEatenNumber
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
# from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpWithNoise, GetCollisionForce, \
    ApplyActionForce, ApplyEnvironForce, getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState, \
    ObserveWithCaughtHistory, ReshapeActionVariousForce, ResetMultiAgentNewtonChasingVariousSheepWithCaughtHistory, \
    CalSheepCaughtHistory, IntegrateStateWithCaughtHistory, RewardWolfWithBiteAndKill, IsCollision, BuildGaussianFixCov, sampleFromContinuousSpace
from collections import OrderedDict

def main():
    dirName = os.path.dirname(__file__)

    maxTrialStep = 360
    wolfActionUpdateInterval = 1
    sheepActionUpdateInterval = 1
    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [4]
    manipulatedVariables['sheepWolfForceRatio'] = [1.0]
    manipulatedVariables['sheepConcern'] = ['all']
    # manipulatedVariables['sheepConcern'] = ['self', 'all']
    trailNumEachCondition = 10

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    # random.shuffle(AllConditions)

    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter players' name:").capitalize()

    mapSize = 1.0
    displaySize = 1.1
    minDistance = mapSize * 1 / 3
    wolfSize = 0.065
    sheepSize = 0.065
    blockSize = 0.13

    screenWidth = int(800)
    screenHeight = int(800)
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    # targetColor = [THECOLORS['orange'], THECOLORS['chocolate1'], THECOLORS['tan1'], THECOLORS['goldenrod2']]
    #'orange', (255, 165, 0); 'chocolate1', (255, 127, 36); 'tan1', (255, 165, 79); 'goldenrod1', (255, 193, 37)
    targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['red3'], THECOLORS['blue3'], THECOLORS['green4']]
    blockColors = [THECOLORS['white']] * 2
    textColorTuple = THECOLORS['green']

    gridSize = 40
    leaveEdgeSpace = 5
    playerRadius = int(wolfSize/(displaySize*2)*screenWidth*gridSize/(gridSize+2*leaveEdgeSpace))
    targetRadius = int(sheepSize/(displaySize*2)*screenWidth*gridSize/(gridSize+2*leaveEdgeSpace))
    blockRadius = int(blockSize/(displaySize*2)*screenWidth*gridSize/(gridSize+2*leaveEdgeSpace))
    stopwatchUnit = 100
    finishTime = 1000 * 15
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
    numBlocks = 2
    sheepLife = 3
    allSheepPolicy = {}
    allWolfPolicy = {}
    allWolfRewardFun = {}
    allTransitFun = {}
    for numSheeps in manipulatedVariables['sheepNums']:
        for sheepConcern in manipulatedVariables['sheepConcern']:
            numAgents = numWolves + numSheeps
            numEntities = numAgents + numBlocks
            wolvesID = list(range(numWolves))
            sheepsID = list(range(numWolves, numAgents))
            blocksID = list(range(numAgents, numEntities))

            entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

            wolfMaxSpeed = 1.0
            sheepMaxSpeed = 1.0
            blockMaxSpeed = None

            entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
            entitiesMovableList = [True] * numAgents + [False] * numBlocks
            massList = [1.0] * numEntities
            reset = ResetMultiAgentNewtonChasingVariousSheepWithCaughtHistory(numWolves, numBlocks, mapSize, minDistance)
            isCollision = IsCollision(getPosFromAgentState)
            rewardWolf = RewardWolfWithBiteAndKill(wolvesID, sheepsID, entitiesSizeList, isCollision, getCaughtHistoryFromAgentState, sheepLife)
            allWolfRewardFun.update({(numSheeps, sheepConcern): rewardWolf})

            stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-displaySize, displaySize], [-displaySize, displaySize])

            def checkBoudary(agentState):
                newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
                return newState

            def checkBoudaryWithCaughtHistory(agentState):
                position = getPosFromAgentState(agentState)
                velocity = getVelFromAgentState(agentState)
                adjustedState = stayInBoundaryByReflectVelocity(position, velocity)
                if len(agentState) == 5:
                    adjustedState = np.append(adjustedState, getCaughtHistoryFromAgentState(agentState))
                return adjustedState.copy()

            # checkAllAgents = lambda states: [checkBoudary(agentState) for agentState in states]
            checkAllAgents = lambda states: [checkBoudaryWithCaughtHistory(agentState) for agentState in states]
            # reshapeHumanAction = ReshapeHumanAction()
            # reshapeSheepAction = ReshapeSheepAction()
            reShapeAction = ReshapeActionVariousForce()
            getCollisionForce = GetCollisionForce()
            applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
            applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                                  getPosFromAgentState)
            calSheepCaughtHistory = CalSheepCaughtHistory(wolvesID, numBlocks, entitiesSizeList, isCollision, sheepLife)
            integrateState = IntegrateStateWithCaughtHistory(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                            getVelFromAgentState, getPosFromAgentState, calSheepCaughtHistory, damping=0.25, dt=0.05)

            actionDimReshaped = 2
            cov = [0.3 ** 2 for _ in range(actionDimReshaped)]
            buildGaussian = BuildGaussianFixCov(cov)
            noiseAction = lambda state: sampleFromContinuousSpace(buildGaussian(tuple(state)))
            transit = TransitMultiAgentChasingForExpWithNoise(reShapeAction, reShapeAction, applyActionForce,
                                                              applyEnvironForce, integrateState, checkAllAgents,
                                                              noiseAction)
            allTransitFun.update({(numSheeps, sheepConcern): transit})

            # transit = TransitMultiAgentChasingForExpVariousForce(reShapeAction, reShapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)

            def loadPolicyOneCondition(numSheeps, sheepConcern):
                # -----------observe--------
                if sheepConcern == 'self':
                    numSheepToObserve = 1
                if sheepConcern == 'all':
                    numSheepToObserve = numSheeps

                observeOneAgent1 = lambda agentID, sId: ObserveWithCaughtHistory(agentID, wolvesID, sId, blocksID,
                                                                                 getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState)
                observeOneAgent = ft.partial(observeOneAgent1, sId=sheepsID)
                observeOne = lambda state, num: [observeOneAgent(agentID)(state) for agentID in range(num)]
                observe = ft.partial(observeOne, num=numAgents)
                initObsForParams = observe(reset(numSheeps))
                obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

                wolvesIDForSheepObserve = list(range(numWolves))
                sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
                blocksIDForSheepObserve = list(
                    range(numSheeps + numWolves, numSheeps + numWolves + numBlocks))
                observeOneAgentForSheep1 = lambda agentID, sId: ObserveWithCaughtHistory(agentID, wolvesIDForSheepObserve, sId,
                                                                        blocksIDForSheepObserve, getPosFromAgentState,
                                                                        getVelFromAgentState, getCaughtHistoryFromAgentState)
                observeOneAgentForSheep = ft.partial(observeOneAgentForSheep1, sId=sheepsIDForSheepObserve)
                observeOneForSheep = lambda state, num: [observeOneAgentForSheep(agentID)(state) for agentID in
                                                         range(num)]
                sheepObserve = ft.partial(observeOneForSheep, num=numWolves + numSheepToObserve)
                sheepObsList = []
                for sheepId in sheepsID:
                    obsFunList = [ObserveWithCaughtHistory(agentID, wolvesIDForSheepObserve, [sheepId], blocksIDForSheepObserve,
                                          getPosFromAgentState, getVelFromAgentState, getCaughtHistoryFromAgentState)
                                  for agentID in list(range(numWolves)) + [sheepId]]
                    sheepObsLambda = lambda state, obsList: list([obs(state) for obs in obsList])
                    sheepObs = ft.partial(sheepObsLambda, obsList=obsFunList)
                    sheepObsList.append(sheepObs)
                initSheepObsForParams = sheepObserve(reset(numSheepToObserve))
                obsSheepShape = [initSheepObsForParams[obsID].shape[0] for obsID in range(len(initSheepObsForParams))]

                worldDim = 2
                actionDim = worldDim * 2 + 1
                layerWidth = [128, 128]

                # -----------model--------
                # modelFolderName = 'withoutWall3wolves'
                # modelFolderName = 'withoutWall2wolves'
                modelFolderName = 'newRewardIndividalAllSheep8Wepisode0.05dt'

                maxEpisode = 80000
                evaluateEpisode = 80000
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
                wolfFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}individ_agent".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                sheepFileNameSep = "maddpg{}wolves1sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent3".format(numWolves, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                # sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0individ1.0_agent".format(numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                sheepFileNameAll = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}individ_agent".format(numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)

                wolfModelPaths = [os.path.join(modelFolder, wolfFileName + str(i) + str(evaluateEpisode) + 'eps') for i in range(numWolves)]
                sheepModelPathsAll = [os.path.join(modelFolder, sheepFileNameAll + str(i) + str(evaluateEpisode) + 'eps') for i in range(numWolves, numWolves + numSheepToObserve)]
                sheepModelPathsSep = [os.path.join(modelFolder, 'trainingId' + str(i) + sheepFileNameSep + str(evaluateEpisode) + 'eps')for i in range(numSheeps)]

                actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)

                [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]
                wolfPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in wolfModelsList]
                wolfPolicyOneCondition = ft.partial(wolfPolicyFun, obs=observe)

                if sheepConcern == 'self':
                    [restoreVariables(model, path) for model, path in zip(sheepModelsListSep, sheepModelPathsSep)]
                    sheepPolicyFun = lambda allAgentsStates: list([actOneStepOneModel(model, sheepObsList[i](allAgentsStates)) for i, model in enumerate(sheepModelsListSep)])
                    sheepPolicyOneCondition = sheepPolicyFun
                else:
                    [restoreVariables(model, path) for model, path in zip(sheepModelsListAll, sheepModelPathsAll)]
                    sheepPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in sheepModelsListAll]
                    sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=sheepObserve)

                return sheepPolicyOneCondition, wolfPolicyOneCondition

            sheepPolicy, wolfPolicy = loadPolicyOneCondition(numSheeps, sheepConcern)
            allSheepPolicy.update({(numSheeps, sheepConcern): sheepPolicy})
            allWolfPolicy.update({(numSheeps, sheepConcern): wolfPolicy})

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone = wolfSize + sheepSize
    recordEaten = RecordEatenNumber(isAnyKilled)
    modelController = allWolfPolicy
    # humanController = JoyStickForceControllers()
    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
    getEntityCaughtHistory = lambda state, entityID: getCaughtHistoryFromAgentState(state[entityID])

    drawImage = DrawImage(screen)
    # drawImageBoth = DrawImageWithJoysticksCheck(screen,humanController.joystickList)
    drawBackground = DrawBackgroundWithStep(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
    drawNewState = DrawNewStateWithBlocksAndFeedback(screen, drawBackground, playerColors, blockColors, targetRadius,
                                                     playerRadius, blockRadius, displaySize, sheepLife)
    trial = NewtonChaseTrialAllCondtionVariouSpeedForModel(screen, killzone, targetColor, numWolves, numBlocks,
                                                           stopwatchEvent, drawNewState, recordEaten, modelController,
                                                           getEntityPos, getEntityVel, getEntityCaughtHistory,
                                                           allSheepPolicy, allTransitFun, allWolfRewardFun, maxTrialStep,
                                                           wolfActionUpdateInterval, sheepActionUpdateInterval)

    hasRest = False

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
