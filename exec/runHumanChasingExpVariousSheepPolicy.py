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
from src.visualization import DrawBackground, DrawImage, GiveExperimentFeedback, InitializeScreen, \
    DrawAttributionTrail, DrawImageWithJoysticksCheck, DrawNewStateWithBlocks
from src.controller import HumanController, ModelController, JoyStickForceControllers
from src.writer import WriteDataFrameToCSV
from src.trial import NewtonChaseTrialAllCondtionVariouSpeed, isAnyKilled, CheckTerminationOfTrial, RecordEatenNumber
from src.experiment import NewtonExperiment
from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables, GetSavePath
from env.multiAgentEnv import StayInBoundaryByReflectVelocity, TransitMultiAgentChasingForExpWithNoise, GetCollisionForce, ApplyActionForce, ApplyEnvironForce, \
    IntegrateState, getPosFromAgentState, getVelFromAgentState, Observe, ReshapeActionVariousForce ,ResetMultiAgentNewtonChasingVariousSheep,  BuildGaussianFixCov, sampleFromContinuousSpace
from collections import OrderedDict

def main():
    dirName = os.path.dirname(__file__)

    manipulatedVariables = OrderedDict()
    manipulatedVariables['sheepNums'] = [1, 2, 4]
    manipulatedVariables['sheepWolfForceRatio'] = [1.3]
    manipulatedVariables['sheepConcern'] = ['selfSheep', 'allSheep']
    trailNumEachCondition = 30

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    AllConditions = parametersAllCondtion * trailNumEachCondition
    random.shuffle(AllConditions)

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
    playerRadius = int(wolfSize / (displaySize * 2) * screenWidth * gridSize / (gridSize + 2 * leaveEdgeSpace))
    targetRadius = int(sheepSize / (displaySize * 2) * screenWidth * gridSize / (gridSize + 2 * leaveEdgeSpace))
    blockRadius = int(blockSize / (displaySize * 2) * screenWidth * gridSize / (gridSize + 2 * leaveEdgeSpace))
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
    writer = WriteDataFrameToCSV(writerPath)
    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction-waitall.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest-waitall.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    # finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, textColorTuple, playerColors)
    drawNewState = DrawNewStateWithBlocks(screen, drawBackground, targetColor, playerColors, blockColors, targetRadius, playerRadius, blockRadius, displaySize)
    drawImage = DrawImage(screen)

    # --------environment setting-----------
    numWolves = 2
    experimentValues["numWolves"] = numWolves
    numBlocks = 0
    allSheepPolicy = {}

    for numSheeps in manipulatedVariables['sheepNums']:
        for sheepConcern in manipulatedVariables['sheepConcern']:
            numAgents = numWolves + numSheeps
            numEntities = numAgents + numBlocks
            wolvesID = list(range(numWolves))
            sheepsID = list(range(numWolves, numAgents))
            blocksID = list(range(numAgents, numEntities))

            entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [blockSize] * numBlocks

            wolfMaxSpeed = 1
            sheepMaxSpeed = 1.3
            blockMaxSpeed = None

            entityMaxSpeedList = [wolfMaxSpeed] * numWolves + [sheepMaxSpeed] * numSheeps + [blockMaxSpeed] * numBlocks
            entitiesMovableList = [True] * numAgents + [False] * numBlocks
            massList = [1.0] * numEntities
            reset = ResetMultiAgentNewtonChasingVariousSheep(numWolves, numBlocks, mapSize, minDistance)
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

            actionDimReshaped = 2
            cov = [3 ** 2 for _ in range(actionDimReshaped)]
            buildGaussian = BuildGaussianFixCov(cov)
            noiseAction = lambda state: sampleFromContinuousSpace(buildGaussian(tuple(state)))
            transit = TransitMultiAgentChasingForExpWithNoise(reShapeAction, reShapeAction, applyActionForce,
                                                              applyEnvironForce, integrateState, checkAllAgents,
                                                              noiseAction)
            # transit = TransitMultiAgentChasingForExpVariousForce(reShapeAction, reShapeAction, applyActionForce, applyEnvironForce, integrateState, checkAllAgents)

            def loadPolicyOneCondition(numSheeps, numSheepToObserve):
                # -----------observe--------
                wolvesIDForSheepObserve = list(range(numWolves))
                sheepsIDForSheepObserve = list(range(numWolves, numSheepToObserve + numWolves))
                blocksIDForSheepObserve = list(range(numSheepToObserve + numWolves, numSheepToObserve + numWolves + numBlocks))
                observeOneAgentForSheep1 = lambda agentID, sId: Observe(agentID, wolvesIDForSheepObserve, sId, blocksIDForSheepObserve,
                                                                        getPosFromAgentState, getVelFromAgentState)
                observeOneAgentForSheep = ft.partial(observeOneAgentForSheep1, sId=sheepsIDForSheepObserve)
                observeOneForSheep = lambda state, num: [observeOneAgentForSheep(agentID)(state) for agentID in range(num)]
                sheepObserve = ft.partial(observeOneForSheep, num=numWolves + numSheepToObserve)
                initSheepObsForParams = sheepObserve(reset(numSheepToObserve))
                obsSheepShape = [initSheepObsForParams[obsID].shape[0] for obsID in range(len(initSheepObsForParams))]

                worldDim = 2
                actionDim = worldDim * 2 + 1
                layerWidth = [128, 128]

                # -----------model--------
                # modelFolderName = 'retrain3wolves'
                modelFolderName = 'withoutWall2wolves'
                maxEpisode = 60000
                evaluateEpisode = 60000
                maxTimeStep = 75
                modelSheepSpeed = 1.0
                buildSheepMADDPGModels = BuildMADDPGModels(actionDim, numWolves + numSheepToObserve, obsSheepShape)
                sheepModelsList = [buildSheepMADDPGModels(layerWidth, agentID) for agentID in
                                   range(numWolves, numWolves + numSheepToObserve)]

                modelFolder = os.path.join(dirName, '..', 'model', modelFolderName)
               # sheepFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}shared_agent".format(numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                sheepFileName = "maddpg{}wolves{}sheep{}blocks{}episodes{}stepSheepSpeed{}WolfActCost0.0individ0.0_agent".format(
                    numWolves, numSheepToObserve, numBlocks, maxEpisode, maxTimeStep, modelSheepSpeed)
                sheepModelPaths = [os.path.join(modelFolder, sheepFileName + str(i) + str(evaluateEpisode) + 'eps') for i
                                   in range(numWolves, numWolves + numSheepToObserve)]
                [restoreVariables(model, path) for model, path in zip(sheepModelsList, sheepModelPaths)]

                actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)

                if numSheepToObserve == 1:
                    sheepModelsListForSelfSheep = sheepModelsList * numSheeps
                    sortState = lambda state, targetIds: [state[id] for id in targetIds]
                    sheepPolicyFun = lambda allAgentsStates, obs: [
                        actOneStepOneModel(model, obs(sortState(allAgentsStates, wolvesID + [sId]))) for model, sId in
                        zip(sheepModelsListForSelfSheep, sheepsID)]
                else:
                    sheepPolicyFun = lambda allAgentsStates, obs: [actOneStepOneModel(model, obs(allAgentsStates)) for
                                                                   model in sheepModelsList]
                sheepPolicyOneCondition = ft.partial(sheepPolicyFun, obs=sheepObserve)
                return sheepPolicyOneCondition

            if sheepConcern == 'selfSheep':
                numSheepToObserve = 1
            if sheepConcern == 'allSheep':
                numSheepToObserve = numSheeps
            sheepPolicy = loadPolicyOneCondition(numSheeps, numSheepToObserve)
            allSheepPolicy.update({(numSheeps, numSheepToObserve): sheepPolicy})

    checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)
    killzone = wolfSize + sheepSize
    recordEaten = RecordEatenNumber(isAnyKilled)
    humanController = JoyStickForceControllers()
    drawImageBoth = DrawImageWithJoysticksCheck(screen, humanController.joystickList)
    getEntityPos = lambda state, entityID: getPosFromAgentState(state[entityID])
    getEntityVel = lambda state, entityID: getVelFromAgentState(state[entityID])
    trial = NewtonChaseTrialAllCondtionVariouSpeed(screen, killzone, numWolves, numBlocks, stopwatchEvent,
                                                   drawNewState, checkTerminationOfTrial, recordEaten, humanController,
                                                   getEntityPos, getEntityVel, allSheepPolicy, transit)

    hasRest = True
    experiment = NewtonExperiment(restImage,hasRest,trial, writer, experimentValues, reset, drawImageBoth)
    # giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)
    drawImageBoth(introductionImage)
    block = 1
    restTimes = 3  # the number of breaks in an experiment
    for i in range(block):
        experiment(finishTime, AllConditions, restTimes)
        # giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        else:
            drawImageBoth(restImage)


if __name__ == "__main__":
    main()
