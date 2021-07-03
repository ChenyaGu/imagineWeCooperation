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
    DrawAttributionTrail
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrialAllCondtion, AttributionTrail, isAnyKilled, CheckEaten, CheckTerminationOfTrial
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
# from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, ResetMultiAgentChasingWithVariousSheep, \
    TransitMultiAgentChasingForExp, ReshapeHumanAction, ReshapeSheepAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe
from collections import OrderedDict

def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [1, 2, 4]
    trailNumEachCondition = 10

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    random.shuffle(AllConditions)

    gridSize = 40
    leaveEdgeSpace = 5

    screenWidth = 800
    screenHeight = 800
    screenCenter = [screenWidth / 2, screenHeight / 2]
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['blue'], THECOLORS['red']]
    textColorTuple = THECOLORS['green']
    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    playerRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace))
    targetRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace))
    totalBarLength = 100
    barHeight = 20
    stopwatchUnit = 100
    finishTime = 1000 * 15
    stopwatchEvent = pg.USEREVENT + 1

    saveImage = False
    wolfSpeedRatio = 1

    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    pg.key.set_repeat(120, 120)
    picturePath = os.path.abspath(os.path.join(os.path.join(dirName, '..'), 'pictures'))
    # resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))

    resultsDicPath = os.path.join(dirName, '..', 'results')
    # resultsDicPath = posixpath.join(dirName, '..', 'results')

    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter players' name:").capitalize()
    experimentValues["name"] = 'test'
    writerPath = os.path.join(resultsDicPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    # finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    # drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

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

        reset = ResetMultiAgentChasingWithVariousSheep(numWolves, numBlocks)
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([-1, 1], [-1, 1])

        def checkBoudary(agentState):
            newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
            return newState

        checkAllAgents = lambda states: [checkBoudary(agentState) for agentState in states]
        reshapeHumanAction = ReshapeHumanAction()
        reshapeSheepAction = ReshapeSheepAction()
        getCollisionForce = GetCollisionForce()
        applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
        applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                              getPosFromAgentState)
        integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                        getVelFromAgentState, getPosFromAgentState)
        transit = TransitMultiAgentChasingForExp(reshapeHumanAction, reshapeSheepAction, applyActionForce, applyEnvironForce, integrateState,
                                                 checkAllAgents)

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
            modelFolderName = os.path.join('fakeNewtonPolicy0618', 'individual={}'.format(individualReward))
            modelSaveName = '{}w{}s'.format(numWolves, numSheeps)
            maxEpisode = 120000
            evaluateEpisode = 120000
            maxTimeStep = 75
            buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
            sheepModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves, numAgents)]
            wolfModelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves)]
            # modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgents)]

            mainModelFolder = os.path.join(dirName, '..', 'model', modelFolderName)
            modelFolder = os.path.join(mainModelFolder, modelSaveName)
            fileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}individ{}_agent".format(numWolves, numSheeps,
                                                                                         numBlocks,
                                                                                         maxEpisode, maxTimeStep,
                                                                                         individualReward)
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

        sheepPolicy = loadPolicyOneCondition(numSheeps)[0]
        allSheepPolicy.update({numSheeps: sheepPolicy})
        wolfPolicy = loadPolicyOneCondition(numSheeps)[1]
        allWolfPolicy.update({numSheeps: wolfPolicy})

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone1 = wolfSize + sheepSize
    killzone2 = (wolfSize + sheepSize) * 0.9
    checkEaten1 = CheckEaten(killzone1, isAnyKilled)
    checkEaten2 = CheckEaten(killzone2, isAnyKilled)
    # attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)
    # sheepPolicy = RandomNewtonMovePolicy(numWolves)
    modelController = allWolfPolicy
    # humanController = JoyStickForceControllers()

    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
    trial1 = NewtonChaseTrialAllCondtion(screen, numWolves, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten1,
                             modelController, getEntityPos, getEntityVel, allSheepPolicy, transit)
    experiment = NewtonExperiment(trial1, writer, experimentValues, reset, drawImage)
    # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    drawImage(introductionImage)
    block = 1

    for i in range(block):
        trialIndex = 0
        score = np.array([0, 0])
        experiment(finishTime, AllConditions)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        else:
            drawImage(restImage)  # 30 sec then press space

    # print(participantsScore)


if __name__ == "__main__":
    main()
