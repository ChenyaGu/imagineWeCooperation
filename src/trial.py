import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
from src.visualization import DrawBackground, DrawNewState, DrawImage, drawText
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld
from pygame.color import THECOLORS
import os
import random


class NewtonChaseTrialAllCondtionVariouSpeedForModel():
    def __init__(self, screen, killzone, targetColors, numOfWolves, numOfBlocks, stopwatchEvent, drawNewState,
                 checkTerminationOfTrial, recordEaten, modelController, getEntityPos, getEntityVel, allSheepPolicy, transit):
        self.screen = screen
        self.killzone = killzone
        self.targetColors = targetColors
        self.numOfWolves = numOfWolves
        self.numOfBlocks = numOfBlocks
        self.modelController = modelController
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.recordEaten = recordEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        self.stopwatchUnit = 100
        self.getEntityPos = getEntityPos
        self.getEntityVel = getEntityVel
        self.allSheepPolicy = allSheepPolicy
        self.transit = transit

    def __call__(self, initState, score, finishTime, currentStopwatch, trialIndex, condition):
        sheepNums = condition['sheepNums']
        sheepConcern = condition['sheepConcern']
        killZone = self.killzone
        targetColors = random.sample(self.targetColors, 4)
        wolfForce = 5
        sheepForce = wolfForce * condition['sheepWolfForceRatio']

        results = co.OrderedDict()
        results["sheepConcern"] = condition['sheepConcern']

        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])
        getPlayerPos = lambda state: [self.getEntityPos(state, agentId) for agentId in range(self.numOfWolves)]
        getTargetPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                      range(self.numOfWolves, self.numOfWolves + sheepNums)]
        getBlockPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                     range(self.numOfWolves + sheepNums, self.numOfBlocks + self.numOfWolves + sheepNums)]
        pause = True
        state = initState
        stateList = []
        initTargetPositions = getTargetPos(initState)
        initPlayerPositions = getPlayerPos(initState)
        initBlockPositions = getBlockPos(initState)
        if initBlockPositions:
            results["blockPositions"] = str(initBlockPositions)
        readyTime = 1000
        currentEatenFlag = [0] * len(initTargetPositions)
        while readyTime > 0:
            pg.time.delay(32)
            self.drawNewState(targetColors, initTargetPositions, initPlayerPositions, initBlockPositions, finishTime, score, currentEatenFlag)
            drawText(self.screen, 'ready', THECOLORS['white'],
                     (self.screen.get_width() * 8 / 3, self.screen.get_height() / 2), 100)
            pg.display.update()
            readyTime -= self.stopwatchUnit
        initialTime = time.get_ticks()
        eatenFlag = [0] * len(initTargetPositions)
        hunterFlag = score
        while pause:
            pg.time.delay(32)
            remainningTime = max(0, finishTime - currentStopwatch)
            targetPositions = getTargetPos(state)
            playerPositions = getPlayerPos(state)
            # wolfPolicy = self.modelController[sheepNums]
            wolfPolicy = self.modelController[sheepNums, sheepConcern]
            humanAction = wolfPolicy(state)
            # sheepPolicy = self.allSheepPolicy[sheepNums]
            sheepPolicy = self.allSheepPolicy[sheepNums, sheepConcern]
            sheepAction = sheepPolicy(state)
            nextState = self.transit(state, humanAction, sheepAction, wolfForce, sheepForce)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    currentStopwatch = currentStopwatch + self.stopwatchUnit
            currentEatenFlag, eatenFlag, hunterFlag = self.recordEaten(targetPositions, playerPositions, killZone, eatenFlag, hunterFlag)
            score = hunterFlag
            self.drawNewState(targetColors, targetPositions, playerPositions, initBlockPositions, remainningTime, score, currentEatenFlag)
            pg.display.update()
            state = nextState
            stateList.append(nextState)
            pause = self.checkTerminationOfTrial(currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results["trialTime"] = wholeResponseTime
        results["hunterFlag"] = str(hunterFlag)
        results["sheepEatenFlag"] = str(eatenFlag)
        results["trialScore"] = sum(eatenFlag)
        wolf1Traj = []
        # wolf1Vel = []
        wolf2Traj = []
        # wolf2Vel = []
        if self.numOfWolves == 3:
            wolf3Traj = []
            # wolf3Vel = []
        sheepTraj = []
        # sheepVel = []
        for i in range(len(stateList)):
            allAgentTraj = stateList[i]
            wolf1Traj.append(self.getEntityPos(allAgentTraj, 0))
            # wolf1Vel.append(self.getEntityVel(allAgentTraj, 0))
            wolf2Traj.append(self.getEntityPos(allAgentTraj, 1))
            # wolf2Vel.append(self.getEntityVel(allAgentTraj, 1))
            if self.numOfWolves == 3:
                wolf3Traj.append(self.getEntityPos(allAgentTraj, 2))
                # wolf3Vel.append(self.getEntityVel(allAgentTraj, 2))
            for j in range(sheepNums):
                sheepTraj.append(self.getEntityPos(allAgentTraj, j + self.numOfWolves))
                # sheepVel.append(self.getEntityVel(allAgentTraj, j + self.numOfWolves))

        roundFunc = lambda x: round(x, 2)
        wolf1Traj = [list(map(roundFunc, i)) for i in wolf1Traj]
        wolf2Traj = [list(map(roundFunc, i)) for i in wolf2Traj]
        sheepTraj = [list(map(roundFunc, i)) for i in sheepTraj]
        # wolf1Vel = [list(map(roundFunc, i)) for i in wolf1Vel]
        # wolf2Vel = [list(map(roundFunc, i)) for i in wolf2Vel]
        # sheepVel = [list(map(roundFunc, i)) for i in sheepVel]

        results["player1 traj"] = str(wolf1Traj)
        results["player2 traj"] = str(wolf2Traj)
        if self.numOfWolves == 3:
            wolf3Traj = [list(map(roundFunc, i)) for i in wolf3Traj]
            # wolf3Vel = [list(map(roundFunc, i)) for i in wolf3Vel]
            results["player3 traj"] = str(wolf3Traj)
            # results["player3 vel"] = str(wolf3Vel)
        results["sheeps traj"] = str(sheepTraj)
        # results["player1 vel"] = str(wolf1Vel)
        # results["player2 vel"] = str(wolf2Vel)
        # results["sheeps vel"] = str(sheepVel)

        totalScore = np.sum(score)
        print(totalScore)
        return results, nextState, score, totalScore, currentStopwatch, eatenFlag


class NewtonChaseTrialAllCondtionVariouSpeed():
    def __init__(self, screen, killzone, targetColors, numOfWolves, numOfBlocks, stopwatchEvent, drawNewState,
                 checkTerminationOfTrial, recordEaten, humanController, getEntityPos, getEntityVel, allSheepPolicy, transit):
        self.screen = screen
        self.killzone = killzone
        self.targetColors = targetColors
        self.numOfWolves = numOfWolves
        self.numOfBlocks = numOfBlocks
        self.humanController = humanController
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.recordEaten = recordEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        self.stopwatchUnit = 100
        self.getEntityPos = getEntityPos
        self.getEntityVel = getEntityVel
        self.allSheepPolicy = allSheepPolicy
        self.transit = transit

    def __call__(self, initState, score, finishTime, currentStopwatch, trialIndex, condition):
        sheepNums = condition['sheepNums']
        sheepConcern = condition['sheepConcern']
        killZone = self.killzone
        targetColors = random.sample(self.targetColors, 4)
        wolfForce = 5
        sheepForce = wolfForce * condition['sheepWolfForceRatio']

        results = co.OrderedDict()

        pickleResults = co.OrderedDict()
        pickleResults['condition'] = condition
        results["sheepConcern"] = condition['sheepConcern']

        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])
        getPlayerPos = lambda state: [self.getEntityPos(state, agentId) for agentId in range(self.numOfWolves)]
        getTargetPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                      range(self.numOfWolves, self.numOfWolves + sheepNums)]
        getBlockPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                     range(self.numOfWolves + sheepNums, self.numOfBlocks + self.numOfWolves + sheepNums)]
        pause = True
        state = initState
        stateList = []
        trajectory = []
        initTargetPositions = getTargetPos(initState)
        initPlayerPositions = getPlayerPos(initState)
        initBlockPositions = getBlockPos(initState)
        if initBlockPositions:
            results["blockPositions"] = str(initBlockPositions)
        readyTime = 1000
        currentEatenFlag = [0] * len(initTargetPositions)
        while readyTime > 0:
            pg.time.delay(32)
            self.drawNewState(targetColors, initTargetPositions, initPlayerPositions, initBlockPositions, finishTime, score, currentEatenFlag)
            drawText(self.screen, 'ready', THECOLORS['white'],
                     (self.screen.get_width() / 8 * 3, self.screen.get_height() / 2), 100)
            pg.display.update()
            readyTime -= self.stopwatchUnit
        initialTime = time.get_ticks()
        eatenFlag = [0] * len(initTargetPositions)
        hunterFlag = score
        while pause:
            pg.time.delay(32)
            remainningTime = max(0, finishTime - currentStopwatch)
            targetPositions = getTargetPos(state)
            playerPositions = getPlayerPos(state)
            humanAction = self.humanController()
            sheepPolicy = self.allSheepPolicy[sheepNums, sheepConcern]
            sheepAction = sheepPolicy(state)
            nextState = self.transit(state, humanAction, sheepAction, wolfForce, sheepForce)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    currentStopwatch = currentStopwatch + self.stopwatchUnit
            currentEatenFlag,eatenFlag, hunterFlag = self.recordEaten(targetPositions, playerPositions, killZone, eatenFlag, hunterFlag)
            score = hunterFlag
            self.drawNewState(targetColors, targetPositions, playerPositions, initBlockPositions, remainningTime, score,currentEatenFlag)
            pg.display.update()
            action = humanAction + sheepAction
            trajectory.append((state, action, nextState))
            state = nextState
            stateList.append(nextState)
            pause = self.checkTerminationOfTrial(currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])
        pickleResults['trajectory'] = trajectory
        results["trialTime"] = wholeResponseTime
        results["hunterFlag"] = str(hunterFlag)
        results["sheepEatenFlag"] = str(eatenFlag)
        results["trialScore"] = sum(eatenFlag)
        wolf1Traj = []
        # wolf1Vel = []
        wolf2Traj = []
        # wolf2Vel = []
        if self.numOfWolves == 3:
            wolf3Traj = []
            # wolf3Vel = []
        sheepTraj = []
        # sheepVel = []
        for i in range(len(stateList)):
            allAgentTraj = stateList[i]
            wolf1Traj.append(self.getEntityPos(allAgentTraj, 0))
            # wolf1Vel.append(self.getEntityVel(allAgentTraj, 0))
            wolf2Traj.append(self.getEntityPos(allAgentTraj, 1))
            # wolf2Vel.append(self.getEntityVel(allAgentTraj, 1))
            if self.numOfWolves == 3:
                wolf3Traj.append(self.getEntityPos(allAgentTraj, 2))
                # wolf3Vel.append(self.getEntityVel(allAgentTraj, 2))
            for j in range(sheepNums):
                sheepTraj.append(self.getEntityPos(allAgentTraj, j + self.numOfWolves))
                # sheepVel.append(self.getEntityVel(allAgentTraj, j + self.numOfWolves))

        roundFunc = lambda x: round(x, 2)
        wolf1Traj = [list(map(roundFunc, i)) for i in wolf1Traj]
        wolf2Traj = [list(map(roundFunc, i)) for i in wolf2Traj]
        sheepTraj = [list(map(roundFunc, i)) for i in sheepTraj]
        # wolf1Vel = [list(map(roundFunc, i)) for i in wolf1Vel]
        # wolf2Vel = [list(map(roundFunc, i)) for i in wolf2Vel]
        # sheepVel = [list(map(roundFunc, i)) for i in sheepVel]

        results["player1 traj"] = str(wolf1Traj)
        results["player2 traj"] = str(wolf2Traj)
        if self.numOfWolves == 3:
            wolf3Traj = [list(map(roundFunc, i)) for i in wolf3Traj]
            # wolf3Vel = [list(map(roundFunc, i)) for i in wolf3Vel]
            results["player3 traj"] = str(wolf3Traj)
            # results["player3 vel"] = str(wolf3Vel)
        results["sheeps traj"] = str(sheepTraj)
        # results["player1 vel"] = str(wolf1Vel)
        # results["player2 vel"] = str(wolf2Vel)
        # results["sheeps vel"] = str(sheepVel)

        totalScore = np.sum(score)
        print(totalScore)
        return pickleResults,results, nextState, score, totalScore, currentStopwatch, eatenFlag


class NewtonChaseTrialAllCondtion():
    def __init__(self, screen, numOfWolves, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,
                 humanController, getEntityPos, getEntityVel, allSheepPolicy, transit):
        self.screen = screen
        self.numOfWolves = numOfWolves
        self.humanController = humanController
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 0.5
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        # self.memorySize = 25
        self.stopwatchUnit = 100
        self.getEntityPos = getEntityPos
        self.getEntityVel = getEntityVel
        self.allSheepPolicy = allSheepPolicy
        self.transit = transit

    def __call__(self, initState, score, finishTime, currentStopwatch, trialIndex, timeStepforDraw, condition):
        sheepNums = condition['sheepNums']
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])
        getPlayerPos = lambda state: [self.getEntityPos(state, agentId) for agentId in range(self.numOfWolves)]
        getTargetPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                      range(self.numOfWolves, self.numOfWolves + sheepNums)]
        # from collections import deque
        # dequeState = deque(maxlen=self.memorySize)
        pause = True
        state = initState
        currentScore = score
        newStopwatch = currentStopwatch
        stateList = []
        initTargetPositions = getTargetPos(initState)
        initPlayerPositions = getPlayerPos(initState)
        readyTime = 1500
        while readyTime > 0:
            pg.time.wait(64)
            self.drawNewState(initTargetPositions, initPlayerPositions, readyTime, currentScore)
            drawText(self.screen, 'ready', THECOLORS['white'],
                     (self.screen.get_width() / 8 * 3, self.screen.get_height() / 2), 100)
            pg.display.update()
            readyTime -= self.stopwatchUnit
        initialTime = time.get_ticks()
        while pause:
            pg.time.wait(64)
            # dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            # targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
            #     targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
            #     sheepNums)
            remainningTime = max(0, finishTime - newStopwatch)
            targetPositions = getTargetPos(state)
            playerPositions = getPlayerPos(state)
            self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)
            pg.display.update()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
            currentStopwatch = newStopwatch
            # wolfPolicy=self.humanController[sheepNums]
            # humanAction=wolfPolicy(state)
            humanAction = self.humanController()
            # action1 = np.array(humanAction[0]) * self.wolfSpeedRatio
            # action2 = np.array(humanAction[1]) * self.wolfSpeedRatio
            # sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, np.array(dequeState) * 10))) / 10 for i in range(sheepNums) ]
            sheepPolicy = self.allSheepPolicy[sheepNums]
            sheepAction = sheepPolicy(state)
            # action = humanAction + sheepAction
            nextState = self.transit(state, humanAction, sheepAction)
            # targetPositions=[self.stayInBoundary(np.add(targetPosition, singleAction))  for (targetPosition,singleAction)  in zip(targetPositions,sheepAction)]
            # playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]
            state = nextState
            stateList.append(nextState)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            pause = self.checkTerminationOfTrial(eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        wolf1Traj = []
        wolf1Vel = []
        wolf2Traj = []
        wolf2Vel = []
        sheepTraj = []
        sheepVel = []
        for i in range(len(stateList)):
            allAgentTraj = stateList[i]
            wolf1Traj.append(self.getEntityPos(allAgentTraj, 0))
            wolf1Vel.append(self.getEntityVel(allAgentTraj, 0))
            wolf2Traj.append(self.getEntityPos(allAgentTraj, 1))
            wolf2Vel.append(self.getEntityVel(allAgentTraj, 1))
            for j in range(sheepNums):
                sheepTraj.append(self.getEntityPos(allAgentTraj, j + 2))
                sheepVel.append(self.getEntityVel(allAgentTraj, j + 2))

        roundFunc = lambda x: round(x, 2)
        wolf1Traj = [list(map(roundFunc, i)) for i in wolf1Traj]
        wolf2Traj = [list(map(roundFunc, i)) for i in wolf2Traj]
        sheepTraj = [list(map(roundFunc, i)) for i in sheepTraj]
        wolf1Vel = [list(map(roundFunc, i)) for i in wolf1Vel]
        wolf2Vel = [list(map(roundFunc, i)) for i in wolf2Vel]
        sheepVel = [list(map(roundFunc, i)) for i in sheepVel]

        results["player1 traj"] = str(wolf1Traj)
        results["player2 traj"] = str(wolf2Traj)
        results["sheeps traj"] = str(sheepTraj)
        results["player1 vel"] = str(wolf1Vel)
        results["player2 vel"] = str(wolf2Vel)
        results["sheeps vel"] = str(sheepVel)
        addScore = [0, 0]
        if True in eatenFlag[:2]:
            # addScore, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["sheepEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward * remainningTime / 1000
        elif True in eatenFlag:
            results["sheepEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward * remainningTime / 1000

        else:
            results["sheepEaten"] = 0
        score = np.add(score, addScore)
        totalScore = score[0] + score[1]
        print(totalScore)
        return results, nextState, score, currentStopwatch, eatenFlag, timeStepforDraw


class NewtonChaseTrial():
    def __init__(self, screen, numOfWolves, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,
                 humanController, getEntityPos, getEntityVel, sheepPolicy, transit):
        self.screen = screen
        self.numOfWolves = numOfWolves
        self.humanController = humanController
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 0.5
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        # self.memorySize = 25
        self.stopwatchUnit = 100
        self.getEntityPos = getEntityPos
        self.getEntityVel = getEntityVel
        self.sheepPolicy = sheepPolicy
        self.transit = transit

    def __call__(self, initState, score, finishTime, currentStopwatch, trialIndex, timeStepforDraw, sheepNums):
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])
        getPlayerPos = lambda state: [self.getEntityPos(state, agentId) for agentId in range(self.numOfWolves)]
        getTargetPos = lambda state: [self.getEntityPos(state, agentId) for agentId in
                                      range(self.numOfWolves, self.numOfWolves + sheepNums)]
        # from collections import deque
        # dequeState = deque(maxlen=self.memorySize)
        pause = True
        state = initState
        currentScore = score
        newStopwatch = currentStopwatch
        stateList = []
        initTargetPositions = getTargetPos(initState)
        initPlayerPositions = getPlayerPos(initState)
        readyTime = 1000
        while readyTime > 0:
            pg.time.delay(32)
            self.drawNewState(initTargetPositions, initPlayerPositions, readyTime, currentScore)
            drawText(self.screen, 'ready', THECOLORS['white'],
                     (self.screen.get_width() / 8 * 3, self.screen.get_height() / 2), 100)
            pg.display.update()
            readyTime -= self.stopwatchUnit
        initialTime = time.get_ticks()
        while pause:
            pg.time.delay(32)
            # dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            # targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
            #     targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
            #     sheepNums)
            remainningTime = max(0, finishTime - newStopwatch)
            targetPositions = getTargetPos(state)
            playerPositions = getPlayerPos(state)
            self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)
            pg.display.update()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
            currentStopwatch = newStopwatch
            humanAction = self.humanController()
            # action1 = np.array(humanAction[0]) * self.wolfSpeedRatio
            # action2 = np.array(humanAction[1]) * self.wolfSpeedRatio
            # sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, np.array(dequeState) * 10))) / 10 for i in range(sheepNums) ]
            sheepAction = self.sheepPolicy(state)
            action = humanAction + sheepAction
            nextState = self.transit(state, action)
            # targetPositions=[self.stayInBoundary(np.add(targetPosition, singleAction))  for (targetPosition,singleAction)  in zip(targetPositions,sheepAction)]
            # playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]
            state = nextState
            stateList.append(nextState)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            pause = self.checkTerminationOfTrial(eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        wolf1Traj = []
        wolf1Vel = []
        wolf2Traj = []
        wolf2Vel = []
        sheepTraj = []
        sheepVel = []
        for i in range(len(stateList)):
            allAgentTraj = stateList[i]
            wolf1Traj.append(self.getEntityPos(allAgentTraj, 0))
            wolf1Vel.append(self.getEntityVel(allAgentTraj, 0))
            wolf2Traj.append(self.getEntityPos(allAgentTraj, 1))
            wolf2Vel.append(self.getEntityVel(allAgentTraj, 1))
            for j in range(sheepNums):
                sheepTraj.append(self.getEntityPos(allAgentTraj, j + 2))
                sheepVel.append(self.getEntityVel(allAgentTraj, j + 2))

        roundFunc = lambda x: round(x, 2)
        wolf1Traj = [list(map(roundFunc, i)) for i in wolf1Traj]
        wolf2Traj = [list(map(roundFunc, i)) for i in wolf2Traj]
        sheepTraj = [list(map(roundFunc, i)) for i in sheepTraj]
        wolf1Vel = [list(map(roundFunc, i)) for i in wolf1Vel]
        wolf2Vel = [list(map(roundFunc, i)) for i in wolf2Vel]
        sheepVel = [list(map(roundFunc, i)) for i in sheepVel]

        results["player1 traj"] = str(wolf1Traj)
        results["player2 traj"] = str(wolf2Traj)
        results["sheeps traj"] = str(sheepTraj)
        results["player1 vel"] = str(wolf1Vel)
        results["player2 vel"] = str(wolf2Vel)
        results["sheeps vel"] = str(sheepVel)
        addScore = [0, 0]
        if True in eatenFlag[:2]:
            # addScore, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["sheepEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward * remainningTime / 1000
        elif True in eatenFlag:
            results["sheepEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward * remainningTime / 1000

        else:
            results["sheepEaten"] = 0
        score = np.add(score, addScore)
        totalScore = score[0] + score[1]
        print(totalScore)
        return results, nextState, score, currentStopwatch, eatenFlag, timeStepforDraw


class AttributionTrail:
    def __init__(self, totalScore, saveImageDir, saveImage, drawAttributionTrail):
        self.totalScore = totalScore
        self.actionDict = [{pg.K_LEFT: -1, pg.K_RIGHT: 1}, {pg.K_a: -1, pg.K_d: 1}]
        self.comfirmDict = [pg.K_RETURN, pg.K_SPACE]
        self.distributeUnit = 0.1
        self.drawAttributionTrail = drawAttributionTrail
        self.saveImageDir = saveImageDir
        self.saveImage = saveImage

    def __call__(self, eatenFlag, hunterFlag, timeStepforDraw):
        hunterid = hunterFlag.index(True)
        attributionScore = [0, 0]
        attributorPercent = 0.5
        pause = True
        screen = self.drawAttributionTrail(hunterid, attributorPercent)
        pg.event.set_allowed([pg.KEYDOWN])

        attributionDelta = 0
        stayAttributionBoudray = lambda attributorPercent: max(min(attributorPercent, 1), 0)
        while pause:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.KEYDOWN:
                    print(event.key)
                    if event.key in self.actionDict[hunterid].keys():
                        attributionDelta = self.actionDict[hunterid][event.key] * self.distributeUnit
                        print(attributionDelta)

                        attributorPercent = stayAttributionBoudray(attributorPercent + attributionDelta)

                        screen = self.drawAttributionTrail(hunterid, attributorPercent)
                    elif event.key == self.comfirmDict[hunterid]:
                        pause = False
            pg.time.wait(10)
            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(screen, self.saveImageDir + '/' + format(timeStepforDraw, '04') + ".png")
                timeStepforDraw += 1

        recipentPercent = 1 - attributorPercent
        if hunterid == 0:
            attributionScore = [int(self.totalScore * attributorPercent), int(self.totalScore * recipentPercent)]
        else:  # hunterid=1
            attributionScore = [int(self.totalScore * recipentPercent), int(self.totalScore * attributorPercent)]

        return attributionScore, timeStepforDraw


def calculateGridDistance(gridA, gridB):
    return np.linalg.norm(np.array(gridA) - np.array(gridB), ord=2)


def isAnyKilled(humanGrids, targetGrid, killzone):
    return np.any(np.array([calculateGridDistance(humanGrid, targetGrid) for humanGrid in humanGrids]) < killzone)


class RecordEatenNumber:

    def __init__(self, isAnyKilled):
        self.isAnyKilled = isAnyKilled

    def __call__(self, targetPositions, playerPositions, killzone, eatenFlag, hunterFlag):
        currentEatenFlag = [0] * len(targetPositions)
        for (i, targetPosition) in enumerate(targetPositions):
            if self.isAnyKilled(playerPositions, targetPosition, killzone):
                eatenFlag[i] += 1
                currentEatenFlag[i] = 1
                break
        for (i, playerPosition) in enumerate(playerPositions):
            if self.isAnyKilled(targetPositions, playerPosition, killzone):
                hunterFlag[i] += 1
                hunterReward = True
                break
        return currentEatenFlag,eatenFlag, hunterFlag


class CheckEatenVariousKillzone:
    def __init__(self, isAnyKilled):
        self.isAnyKilled = isAnyKilled

    def __call__(self, targetPositions, playerPositions, killzone):
        eatenFlag = [False] * len(targetPositions)
        hunterFlag = [False] * len(playerPositions)
        for (i, targetPosition) in enumerate(targetPositions):
            if self.isAnyKilled(playerPositions, targetPosition, killzone):
                eatenFlag[i] = True
                break
        for (i, playerPosition) in enumerate(playerPositions):
            if self.isAnyKilled(targetPositions, playerPosition, killzone):
                hunterFlag[i] = True
                break
        return eatenFlag, hunterFlag


class CheckTerminationOfTrial:
    def __init__(self, finishTime):
        self.finishTime = finishTime

    def __call__(self, currentStopwatch):
        if currentStopwatch >= self.finishTime:
            pause = False
        else:
            pause = True
        return pause


class Trial():
    def __init__(self, actionSpace, killzone, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,
                 attributionTrail, humanController):
        self.humanController = humanController
        self.actionSpace = actionSpace
        self.killzone = killzone
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 1
        self.attributionTrail = attributionTrail
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        self.memorySize = 25

    def __call__(self, targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw,
                 sheepNums):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        from collections import deque
        dequeState = deque(maxlen=self.memorySize)
        pause = True
        traj = []
        while pause:
            pg.time.delay(32)
            dequeState.append(
                [np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
                targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
                sheepNums)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            state = {'playerPos': playerPositions, 'targetPositions': targetPositions, 'action': action,
                     'currentStopwatch': currentStopwatch}
            # (playerPositions,targetPositions,action,currentStopwatch)
            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
            traj.append(state)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addScore = [0, 0]
        if True in eatenFlag[:2]:
            # addScore, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addScore)
        return traj, targetPositions, playerPositions, score, currentStopwatch, eatenFlag, timeStepforDraw


class TrialServer():
    def __init__(self, actionSpace, killzone, stopwatchEvent, checkTerminationOfTrial, checkEaten, humanController):
        self.humanController = humanController
        self.actionSpace = actionSpace
        self.killzone = killzone
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 1
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        self.memorySize = 25

    def __call__(self, targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw,
                 sheepNums):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        from collections import deque
        dequeState = deque(maxlen=self.memorySize)
        pause = True
        traj = []
        while pause:
            pg.time.delay(32)
            dequeState.append(
                [np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
                targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
                sheepNums)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            state = {'playerPos': playerPositions, 'targetPositions': targetPositions, 'action': action,
                     'currentStopwatch': currentStopwatch}

            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
            traj.append(state)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addScore = [0, 0]
        if True in eatenFlag[:2]:
            # addScore, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addScore)
        return traj, targetPositions, playerPositions, score, currentStopwatch, eatenFlag, timeStepforDraw


class ChaseTrial():
    def __init__(self, actionSpace, killzone, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,
                 attributionTrail, humanController):
        self.humanController = humanController
        self.actionSpace = actionSpace
        self.killzone = killzone
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 1
        self.attributionTrail = attributionTrail
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        self.memorySize = 25

    def __call__(self, targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw,
                 sheepNums):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        from collections import deque
        dequeState = deque(maxlen=self.memorySize)
        pause = True
        while pause:
            pg.time.delay(32)
            dequeState.append(
                [np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            # targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
            #     targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
            #     sheepNums)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
            humanAction = self.joyStickController()
            action1 = np.array(humanAction[0]) * self.wolfSpeedRatio
            action2 = np.array(humanAction[1]) * self.wolfSpeedRatio
            sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, np.array(dequeState) * 10))) / 10 for i
                           in range(sheepNums)]

            targetPositions = [self.stayInBoundary(np.add(targetPosition, singleAction)) for
                               (targetPosition, singleAction) in zip(targetPositions, sheepAction)]
            playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in
                               zip(playerPositions, [action1, action2])]

            remainningTime = max(0, self.finishTime - newStopwatch)

            screen = self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)
            pg.display.update()
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)

            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addScore = [0, 0]
        if True in eatenFlag[:2]:
            # addScore, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addScore[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addScore)
        return results, targetPositions, playerPositions, score, currentStopwatch, eatenFlag, timeStepforDraw
