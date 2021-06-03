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
from src.sheepPolicy import RandomNewtonMovePolicy, chooseGreedyAction, sampleAction, \
    SoftmaxAction  # GenerateModel, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, ResetMultiAgentNewtonChasing, \
    TransitMultiAgentChasingForExp, ReshapeAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState
from collections import OrderedDict


#
def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [1, 2, 4]
    trailNumEachCondition = 5

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    random.shuffle(AllConditions)


    gridSize = 60
    # bounds = [0, 0, gridSize - 1, gridSize - 1]
    minDistanceForReborn = 10
    numPlayers = 2


    screenWidth = 800
    screenHeight = 800
    screenCenter = [screenWidth / 2, screenHeight / 2]
    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    leaveEdgeSpace = 6
    lineWidth = 1
    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [THECOLORS['orange']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['blue'], THECOLORS['red']]
    wolfSize = 1.5
    sheepSize = 1.5
    blockSize = 2.25
    targetRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace)*sheepSize)
    playerRadius = int(screenWidth/(gridSize+2*leaveEdgeSpace)*wolfSize)
    totalBarLength = 100
    barHeight = 20
    stopwatchUnit = 100
    finishTime = 1000 * 15
    block = 1
    textColorTuple = THECOLORS['green']
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

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth,
                                    textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'),
                                experimentValues["name"])
    ############### environment setting
    numWolves = 2
    numSheeps = max(manipulatedVariables['sheepNums'])
    numBlocks = 0

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))


    # wolfSize = 0.075
    # sheepSize = 0.075
    # blockSize = 0.075
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1000
    sheepMaxSpeed = 1000
    blockMaxSpeed = None

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities


    reset = ResetMultiAgentNewtonChasing(gridSize, numPlayers, minDistanceForReborn)

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
    ############
    # actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    # preyPowerRatio = 3
    # sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    # numActionSpace = len(sheepActionSpace)

    # actionSpaceStill = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    # sheepActionSpaceStill = list(map(tuple, np.array(actionSpaceStill) * preyPowerRatio))
    # numActionSpaceStill = len(sheepActionSpaceStill)

    sheepPolicy = RandomNewtonMovePolicy(numPlayers)

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone = wolfSize+sheepSize+0.5
    checkEaten = CheckEaten(killzone, isAnyKilled)
    totalScore = 10
    attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)
    humanController = JoyStickForceControllers()
    # humanController =lambda: list(np.random.uniform(-1,1,[2,5]))
    # humanController = HumanController(writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, sheepPolicy, chooseGreedyAction)

    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    # actionSpace = list(it.product([0, 1, -1], repeat=2))
    trial = NewtonChaseTrial(numPlayers, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,
                             humanController, getEntityPos, sheepPolicy, transit)
    experiment = NewtonExperiment(trial, writer, experimentValues, reset, drawImage)
    giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)

    drawImage(introductionImage)
    score = [0] * block

    for i in range(block):
        experiment(finishTime, AllConditions)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        # else:
        # drawImage(restImage)

    # participantsScore = np.sum(np.array(score))
    # print(participantsScore)


if __name__ == "__main__":
    main()
