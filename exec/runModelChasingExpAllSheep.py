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
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback, InitializeScreen, \
    DrawAttributionTrail, DrawImageWithJoysticksCheck, DrawNewStateWithBlocks
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrialAllCondtionVariouSpeedForModel, isAnyKilled, CheckTerminationOfTrial, RecordEatenNumber
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
# from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpVariousForce, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe, ReshapeActionVariousForce, ResetMultiAgentNewtonChasingVariousSheep, RewardWolf, IsCollision
from collections import OrderedDict

def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [1,2,4]
    manipulatedVariables['sheepWolfForceRatio'] = [1.3]
    trailNumEachCondition = 10

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
    targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['blue3'], THECOLORS['red3'], THECOLORS['green3']]
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
    # resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))

    resultsDicPath = os.path.join(dirName, '..', 'results')
    # resultsDicPath = posixpath.join(dirName, '..', 'results')

    
    # experimentValues["name"] = '0704'
    writerPath = os.path.join(resultsDicPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction-waitall.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest-waitall.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    # finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
    drawNewState = DrawNewStateWithBlocks(screen, drawBackground, targetColor, playerColors, blockColors, targetRadius, playerRadius, blockRadius, displaySize)
    drawImage = DrawImage(screen)
    # totalBarLength = 100
    # barHeight = 20
    # screenCenter = [screenWidth / 2, screenHeight / 2]
    # drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    # saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

    # --------environment setting-----------
    numWolves = 3
    numBlocks = 0
    allSheepPolicy = {}
    allWolfPolicy = {}
    for numSheeps in manipulatedVariables['sheepNums']:
        numAgents = numWolves + numSheeps
        numEntities = numAgents + numBlocks
        wolvesID = list(range(numWolves))
        sheepsID = list(range(numWolves, numAgents))
        blocksID = list(range(numAgents, numEntities))

        entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

        wolfMaxSpeed = 1
        sheepMaxSpeed = 1.3
        blockMaxSpeed = None

        individualReward = 0.0

        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities
        reset = ResetMultiAgentNewtonChasingVariousSheep(numWolves, numBlocks, mapSize, minDistance)
        # reset = ResetMultiAgentChasingWithVariousSheep(numWolves, numBlocks)
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-displaySize, displaySize], [-displaySize, displaySize])

        def checkBoudary(agentState):
            newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
            return newState

        checkAllAgents = lambda states: [checkBoudary(agentState) for agentState in states]
        # reshapeHumanAction = ReshapeHumanAction()
        # reshapeSheepAction = ReshapeSheepAction()
        reShapeAction = ReshapeActionVariousForce()
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                              getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                        getVelFromAgentState, getPosFromAgentState)
                                        
        transit = TransitMultiAgentChasingForExpVariousForce(reShapeAction, reShapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)
        # transit = TransitMultiAgentChasingForExp(reshapeHumanAction, reshapeSheepAction, applyActionForce,applyEnvironForce, integrateState, checkAllAgents)
        def loadPolicyOneCondition(numSheeps):
            # -----------observe--------
            observeOneAgent1 = lambda agentID, sId: Observe(agentID, wolvesID, sId, blocksID, getPosFromAgentState,
                                                            getVelFromAgentState)
            observeOneAgent = ft.partial(observeOneAgent1, sId=sheepsID)
            observeOne = lambda state, num: [observeOneAgent(agentID)(state) for agentID in range(num)]
            observe = ft.partial(observeOne, num=numAgents)

            initObsForParams = observe(reset(numSheeps))
            obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
            worldDim = 2
            actionDim = worldDim * 2 + 1

            layerWidth = [128, 128]

            # -----------model--------
            modelFolderName = 'retrain2wolves'
            # modelFolderName = 'withoutWall2wolves'
            maxEpisode = 60000
            evaluateEpisode = 60000
            maxTimeStep = 75
            modelSheepSpeed = 1.0
            buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
            sheepModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves, numAgents)]
            wolfModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves)]

            modelFolder = os.path.join(dirName, '..', 'model', modelFolderName)
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
            # fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0individ0.0_agent".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
            sheepModelPaths = [os.path.join(modelFolder, fileName + str(i) + str(evaluateEpisode) + 'eps') for i in
                               range(numWolves, numAgents)]
            wolfModelPaths = [os.path.join(modelFolder, fileName + str(i) + str(evaluateEpisode) + 'eps') for i in
                              range(numWolves)]
            [restoreVariables(model, path) for model, path in zip(sheepModelsList, sheepModelPaths)]
            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]

            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            sheepPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in sheepModelsList]
            sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=observe)
            wolfPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in wolfModelsList]
            wolfPolicyOneCondition = ft.partial(wolfPolicyFun, obs=observe)
            return sheepPolicyOneCondition, wolfPolicyOneCondition

        sheepPolicy, wolfPolicy = loadPolicyOneCondition(numSheeps)
        allSheepPolicy.update({numSheeps: sheepPolicy})
        allWolfPolicy.update({numSheeps: wolfPolicy})

    # collisionReward = 1  # it is 10 in the training environment
    # isCollision = IsCollision(getPosFromAgentState)
    # rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision, collisionReward,individualReward)
    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone = wolfSize + sheepSize
    recordEaten = RecordEatenNumber(isAnyKilled)
    # attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)
    # sheepPolicy = RandomNewtonMovePolicy(numWolves)
    modelController = allWolfPolicy
    # humanController = JoyStickForceControllers()
    # drawImageBoth = DrawImageWithJoysticksCheck(screen,humanController.joystickList)
    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
    trial = NewtonChaseTrialAllCondtionVariouSpeedForModel(screen, killzone, numWolves, numBlocks, stopwatchEvent, drawNewState, checkTerminationOfTrial, recordEaten, modelController, getEntityPos, getEntityVel, allSheepPolicy, transit)

    hasRest = False  # True
    experiment = NewtonExperiment(restImage,hasRest,trial, writer, experimentValues, reset, drawImage)
    # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    # drawImageBoth(introductionImage)
    block = 1
    restDuration = 60
    for i in range(block):
        experiment(finishTime, AllConditions, restDuration)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
            # drawImage(restImage)
        else:
            # humanController.joystickList
            drawImage(restImage)


if __name__ == "__main__":
    main()
