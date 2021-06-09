import os
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import collections as co
import itertools as it
import numpy as np
import random
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback, InitializeScreen, \
    DrawAttributionTrail
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrial, AttributionTrail, isAnyKilled, CheckEaten, CheckTerminationOfTrial
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath
# from src.sheepPolicy import BuildMADDPGModels, ActOneStep, actByPolicyTrainNoisy, RandomNewtonMovePolicy, \
    # chooseGreedyAction, sampleAction, SoftmaxAction, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, ResetMultiAgentNewtonChasing, \
    TransitMultiAgentChasingForExp, ReshapeAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe
from collections import OrderedDict


#
def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [2]
    trailNumEachCondition = 3

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    random.shuffle(AllConditions)

    gridSize = 60
    minDistanceForReborn = 10
    leaveEdgeSpace = 6

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
    wolfSize = 1.5
    sheepSize = 1.5
    blockSize = 1.5
    playerRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace)*wolfSize)
    targetRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace)*sheepSize)
    totalBarLength = 100
    barHeight = 20
    stopwatchUnit = 100
    finishTime = 1000 * 16
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
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'),
                                experimentValues["name"])

    # --------environment setting-----------
    numWolves = 2
    numSheeps = 2
    numBlocks = 0

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1000
    sheepMaxSpeed = 1000
    blockMaxSpeed = None

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities


    reset = ResetMultiAgentNewtonChasing(gridSize, numWolves, minDistanceForReborn)

    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity([0, gridSize-1], [0, gridSize-1])

    def checkBoudary(agentState):
        newState = stayInBoundaryByReflectVelocity(getPosFromAgentState(agentState), getVelFromAgentState(agentState))
        return newState

    checkAllAgents = lambda states: [checkBoudary(agentState) for agentState in states]
    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList, getCollisionForce,
                                          getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList,
                                    getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasingForExp(reshapeAction, applyActionForce, applyEnvironForce, integrateState,
                                             checkAllAgents)

    # -----------observe--------
    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, [], getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

    initObsForParams = observe(reset(numSheeps))
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # -----------model--------
    modelSaveName = '2w2s'
    maxEpisode = 60000
    evaluateEpisode = 60000
    maxTimeStep = 75
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgents, obsShape)
    # [print(agentID) for agentID in range(numAgents)]
    # buildMADDPGModels(layerWidth, 1)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numWolves,numAgents)]

    mainModelFolder = os.path.join(dirName, '..', 'model','faneNewton')
    modelFolder = os.path.join(mainModelFolder, modelSaveName)
    fileName = "maddpg2wolves2sheep0blocks{}episodes{}stepSheepSpeed1WolfActCost0individ0_agent".format(maxEpisode, maxTimeStep)
    modelPaths = [os.path.join(modelFolder, fileName + str(i) + str(evaluateEpisode) + 'eps') for i in range(numWolves,numAgents)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)

    sheepPolicy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
    # sheepPolicy = RandomNewtonMovePolicy(numWolves)


    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone = wolfSize+sheepSize+0.5
    checkEaten = CheckEaten(killzone, isAnyKilled)
    totalScore = 10
    # attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)
    humanController = JoyStickForceControllers()
    # humanController =lambda: list(np.random.uniform(-1,1,[2,5]))
    # humanController = HumanController(writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, sheepPolicy, chooseGreedyAction)

    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
    # actionSpace = list(it.product([0, 1, -1], repeat=2))
    trial = NewtonChaseTrial(numWolves, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,
                             humanController, getEntityPos, getEntityVel, sheepPolicy, transit)
    experiment = NewtonExperiment(trial, writer, experimentValues, reset, drawImage)
    giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    drawImage(introductionImage)

    block = 2
    score = [0] * block

    for i in range(block):
        experiment(finishTime, AllConditions)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        else:
            drawImage(restImage)  # 30 sec then press space

    # participantsScore = np.sum(np.array(score))
    # print(participantsScore)


if __name__ == "__main__":
    main()
