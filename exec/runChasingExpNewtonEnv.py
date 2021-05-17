import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import collections as co
import itertools as it
import numpy as np
import posixpath
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback, InitializeScreen, DrawAttributionTrail
from src.controller import HumanController, ModelController
from src.updateWorld import  UpdateWorld, StayInBoundary
from src.writer import WriteDataFrameToCSV,saveToPickle
from src.trial import Trial, AttributionTrail, isAnyKilled, CheckEaten, CheckTerminationOfTrial
from src.experiment import Experiment
from src.sheepPolicy import RandomMovePolicy, chooseGreedyAction, sampleAction, SoftmaxAction #GenerateModel, restoreVariables, ApproximatePolicy
from env.multiAgentEnv import ResetMultiAgentChasingForExp,TransitMultiAgentChasing, ReshapeAction, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, IntegrateState, getPosFromAgentState,getVelFromAgentState 
from collections import OrderedDict
#
def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums']=[2,4,8]
    trailNumEachCondition = 1

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    AllConditions=parametersAllCondtion*trailNumEachCondition

    numWolves = 2
    numSheeps = 1
    numBlocks = 2


    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks
    wolvesID = list(range(numWolves))
    sheepsID = list(range(numWolves, numAgents))
    blocksID = list(range(numAgents, numEntities))

    wolfSize = 0.075
    sheepSize = 0.05
    blockSize = 0.2
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

    wolfMaxSpeed = 1.0
    blockMaxSpeed = None
    sheepMaxSpeedOriginal = 1.3
    # sheepMaxSpeed = sheepMaxSpeedOriginal * sheepSpeedMultiplier
    sheepMaxSpeed = sheepMaxSpeedOriginal

    entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities
    gridSize = 60
    bounds = [0, 0, gridSize - 1, gridSize - 1]
    minDistanceForReborn = 5
    numPlayers = 2
    reset = ResetMultiAgentChasingForExp(bounds, numPlayers, minDistanceForReborn)

    reshapeAction = ReshapeAction()
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,  getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList, entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    transit = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)
    # minDistanceForReborn
    # updateWorld = UpdateWorld(bounds,  minDistanceForReborn)


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
    targetColor = [THECOLORS['blue']]*16 # [255, 50, 50]
    playerColors = [THECOLORS['orange'], THECOLORS['red']]
    targetRadius = 10
    playerRadius = 10
    totalBarLength = 100
    barHeight = 20
    stopwatchUnit = 100
    # finishTime = 1000 * 60 * 2
    finishTime = 1000 * 15
    block = 1
    softmaxBeita = -1
    textColorTuple = THECOLORS['green']
    stopwatchEvent = pg.USEREVENT + 1

    saveImage = False
    killzone = 2
    wolfSpeedRatio = 1

    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    pg.key.set_repeat(120, 120)
    picturePath = os.path.abspath(os.path.join(os.path.join(dirName, '..'), 'pictures'))
    # resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))

    resultsDicPath = os.path.join(dirName,  '..', 'results')
    # resultsDicPath = posixpath.join(dirName, '..', 'results')

    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["name"] = 'kill' + str(killzone)
    experimentValues["condition"] = 'all'
    writerPath = os.path.join(resultsDicPath, experimentValues["name"]) + '.pickle'
    # writer = WriteDataFrameToCSV(writerPath)
    writer = saveToPickle
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    drawAttributionTrail = DrawAttributionTrail(screen, playerColors, totalBarLength, barHeight, screenCenter)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

    xBoundary = [bounds[0], bounds[2]]
    yBoundary = [bounds[1], bounds[3]]
    stayInBoundary = StayInBoundary(xBoundary, yBoundary)

############
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    numActionSpace = len(sheepActionSpace)

    actionSpaceStill = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    sheepActionSpaceStill = list(map(tuple, np.array(actionSpaceStill) * preyPowerRatio))
    numActionSpaceStill = len(sheepActionSpaceStill)


    sheepPolicy=RandomMovePolicy(sheepActionSpace)

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    checkEaten = CheckEaten(killzone, isAnyKilled)
    totalScore = 10
    attributionTrail = AttributionTrail(totalScore, saveImageDir, saveImage, drawAttributionTrail)

    humanController = HumanController(writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, sheepPolicy, chooseGreedyAction)

    actionSpace = list(it.product([0, 1, -1], repeat=2))
    trial = Trial(actionSpace, killzone, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten, attributionTrail, humanController)
    experiment = Experiment(trial, writer, experimentValues, reset, updateWorld, drawImage, writerPath)
    giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)

    # drawImage(introductionImage)
    score = [0] * block

    for i in range(block):
        score[i] = experiment(finishTime,AllConditions)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        # else:
            # drawImage(restImage)

    participantsScore = np.sum(np.array(score))
    print(participantsScore)


if __name__ == "__main__":
    main()
