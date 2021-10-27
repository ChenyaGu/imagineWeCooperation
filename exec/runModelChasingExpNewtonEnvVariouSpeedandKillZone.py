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
    DrawAttributionTrail, DrawImageWithJoysticksCheck
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrialAllCondtionVariouSpeedAndKillZoneForModel, AttributionTrail, isAnyKilled, CheckEaten, CheckTerminationOfTrial,CheckEatenVariousKillzone
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
# from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, ResetMultiAgentChasingWithVariousSheep, \
    TransitMultiAgentChasingForExpVariousForce, ReshapeHumanAction, ReshapeSheepAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe,ReshapeActionVariousForce,ResetMultiAgentNewtonChasingVariousSheep
from collections import OrderedDict

def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [4]
    manipulatedVariables['sheepWolfForceRatio'] = [1.3]
    manipulatedVariables['killZoneRatio'] = [1.0]
    trailNumEachCondition = 100

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    # random.shuffle(AllConditions)

    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter players' name:").capitalize()

    mapSize = 1.0
    minDistance = mapSize * 1 / 3
    sizeRatio = 1.0
    wolfSize = 0.075 * sizeRatio
    sheepSize = 0.05 * sizeRatio
    blockSize = 0.0

    screenWidth = int(800 * mapSize)
    screenHeight = int(800 * mapSize)
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['blue'], THECOLORS['red']]
    textColorTuple = THECOLORS['green']

    gridSize = 40
    leaveEdgeSpace = 5
    playerRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace))
    targetRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace))

    stopwatchUnit = 100
    finishTime = 1000 * 15
    stopwatchEvent = pg.USEREVENT + 1

    # saveImage = False
    # wolfSpeedRatio = 1

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
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius, mapSize)
    drawImage = DrawImage(screen)
    # totalBarLength = 100
    # barHeight = 20
    # screenCenter = [screenWidth / 2, screenHeight / 2]
    # drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    # saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

    # --------environment setting-----------
    numWolves = 2
    # numSheeps = max(manipulatedVariables['sheepNums'])
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

        individualReward = 1

        entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
        entitiesMovableList = [True] * numAgents + [False] * numBlocks
        massList = [1.0] * numEntities
        reset = ResetMultiAgentNewtonChasingVariousSheep(numWolves, mapSize, minDistance)
        # reset = ResetMultiAgentChasingWithVariousSheep(numWolves, numBlocks)
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-mapSize, mapSize], [-mapSize, mapSize])

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
            # modelFolderName = os.path.join('individualReward={}'.format(individualReward), 'sheepWolfForceRatio={}_killZoneRatio={}'.format(manipulatedVariables['sheepWolfForceRatio'][0], manipulatedVariables['killZoneRatio'][0]))
            # modelFolderName = os.path.join('maxRange{}'.format(mapSize), 'sizeRatio={}'.format(sizeRatio))
            modelFolderName = 'maxTimeStep=100'
            modelSaveName = '{}w{}s'.format(numWolves, numSheeps)
            maxEpisode = 120000
            evaluateEpisode = 92000
            maxTimeStep = 100
            buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
            sheepModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves, numAgents)]
            wolfModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves)]
            # modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

            mainModelFolder = os.path.join(dirName, '..', 'model', modelFolderName)
            modelFolder = os.path.join(mainModelFolder, modelSaveName)
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}individ{}_agent".format(numWolves, numSheeps, numBlocks, maxEpisode, maxTimeStep, individualReward)
            sheepModelPaths = [os.path.join(modelFolder, fileName + str(i) + str(evaluateEpisode) + 'eps') for i in
                               range(numWolves, numAgents)]
            wolfModelPaths = [os.path.join(modelFolder, fileName + str(i) + str(evaluateEpisode) + 'eps') for i in
                              range(numWolves)]
            [restoreVariables(model, path) for model, path in zip(sheepModelsList, sheepModelPaths)]
            [restoreVariables(model, path) for model, path in zip(wolfModelsList, wolfModelPaths)]

            actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
            sheepPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for model in sheepModelsList]
            sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=observe)
            wolfPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, observe(allAgentsStates)) for model in wolfModelsList]
            wolfPolicyOneCondition = ft.partial(wolfPolicyFun, obs=observe)
            return sheepPolicyOneCondition, wolfPolicyOneCondition

        sheepPolicy, wolfPolicy = loadPolicyOneCondition(numSheeps)
        allSheepPolicy.update({numSheeps: sheepPolicy})
        allWolfPolicy.update({numSheeps: wolfPolicy})

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    baselineKillzone = 0  # wolfSize + sheepSize
    checkEaten = CheckEatenVariousKillzone(isAnyKilled)
    # checkEaten = CheckEaten(killzone, isAnyKilled)
    # attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)
    # sheepPolicy = RandomNewtonMovePolicy(numWolves)
    modelController = allWolfPolicy
    # humanController = JoyStickForceControllers()
    # drawImageBoth = DrawImageWithJoysticksCheck(screen,humanController.joystickList)
    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
    trial = NewtonChaseTrialAllCondtionVariouSpeedAndKillZoneForModel(screen,baselineKillzone, numWolves, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten, modelController, getEntityPos, getEntityVel, allSheepPolicy, transit)

    hasRest = False  # True
    experiment = NewtonExperiment(restImage,hasRest,trial, writer, experimentValues, reset, drawImage)
    # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    # drawImageBoth(introductionImage)
    block = 1
    restDuration = 60
    for i in range(block):
        score = np.array([0, 0])
        experiment(finishTime, AllConditions,restDuration)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
            # drawImage(restImage)
        else:
            # humanController.joystickList
            drawImage(restImage)


if __name__ == "__main__":
    main()
