import numpy as np
import pygame as pg
from pygame import time
import collections as co
import pickle
from src.visualization import DrawBackground, DrawNewState, DrawImage, drawText
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld
import os
class NewtonChaseTrial():
    def __init__(self,numOfWolves, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten,humanController,getEntityPos,sheepPolicy,transit):
        self.numOfWolves = numOfWolves
        self.humanController = humanController
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.beanReward = 1
        self.checkEaten = checkEaten
        self.checkTerminationOfTrial = checkTerminationOfTrial
        # self.memorySize = 25
        self.stopwatchUnit = 100
        self.getEntityPos = getEntityPos    
        self.sheepPolicy = sheepPolicy
        self.transit = transit
    def __call__(self,initState, score, finishTime, currentStopwatch, trialIndex, timeStepforDraw, sheepNums):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])
        getPlayerPos = lambda state: [self.getEntityPos(state,agentId) for agentId in range(self.numOfWolves)]
        getTargetPos = lambda state: [self.getEntityPos(state,agentId) for agentId in range(self.numOfWolves,self.numOfWolves + sheepNums)]
        # from collections import deque
        # dequeState = deque(maxlen=self.memorySize)
        pause = True
        state = initState
        currentScore = score
        newStopwatch = currentStopwatch

        while pause:
            pg.time.delay(32)
            # dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            # targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
            #     targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
            #     sheepNums)

            targetPositions = getTargetPos(state)
            playerPositions = getPlayerPos(state)
            remainningTime = max(0, finishTime - newStopwatch)
            self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)
            pg.display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
            currentStopwatch = newStopwatch
            humanAction=self.humanController()
            # action1 = np.array(humanAction[0]) * self.wolfSpeedRatio
            # action2 = np.array(humanAction[1]) * self.wolfSpeedRatio
            # sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, np.array(dequeState) * 10))) / 10 for i in range(sheepNums) ]
            sheepAction = self.sheepPolicy(state)
            action = humanAction + sheepAction
            nextState = self.transit(state,action)
            # targetPositions=[self.stayInBoundary(np.add(targetPosition, singleAction))  for (targetPosition,singleAction)  in zip(targetPositions,sheepAction)]
            # playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]
            state = nextState
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)

            pause = self.checkTerminationOfTrial(eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addSocre = [0, 0]
        if True in eatenFlag[:2]:
            # addSocre, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addSocre)
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


class CheckEaten:
    def __init__(self, killzone, isAnyKilled):
        self.killzone = killzone
        self.isAnyKilled = isAnyKilled

    def __call__(self, targetPositions, playerPositions):
        eatenFlag = [False] * len(targetPositions)
        hunterFlag = [False] * len(playerPositions)
        for (i, targetPosition) in enumerate(targetPositions):
            if self.isAnyKilled(playerPositions, targetPosition, self.killzone):
                eatenFlag[i] = True
                break
        for (i, playerPosition) in enumerate(playerPositions):
            if self.isAnyKilled(targetPositions, playerPosition, self.killzone):
                hunterFlag[i] = True
                break
        return eatenFlag, hunterFlag


class CheckTerminationOfTrial:
    def __init__(self, finishTime):
        self.finishTime = finishTime

    def __call__(self, eatenFlag, currentStopwatch):
        if np.any(eatenFlag) == True or currentStopwatch >= self.finishTime:
            pause = False
        else:
            pause = True
        return pause


class Trial():
    def __init__(self, actionSpace, killzone, stopwatchEvent, drawNewState, checkTerminationOfTrial, checkEaten, attributionTrail, humanController):
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

    def __call__(self, targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw,sheepNums):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        from collections import deque
        dequeState = deque(maxlen=self.memorySize)
        pause = True
        traj=[]
        while pause:
            pg.time.delay(32)
            dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,sheepNums)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            state={'playerPos':playerPositions,'targetPositions':targetPositions,'action':action,'currentStopwatch':currentStopwatch}
            # (playerPositions,targetPositions,action,currentStopwatch)
            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
            traj.append(state)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addSocre = [0, 0]
        if True in eatenFlag[:2]:
            # addSocre, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addSocre)
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

    def __call__(self, targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw,sheepNums):
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        from collections import deque
        dequeState = deque(maxlen=self.memorySize)
        pause = True
        traj=[]
        while pause:
            pg.time.delay(32)
            dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,sheepNums)
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
            state={'playerPos':playerPositions,'targetPositions':targetPositions,'action':action,'currentStopwatch':currentStopwatch}

            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
            traj.append(state)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addSocre = [0, 0]
        if True in eatenFlag[:2]:
            # addSocre, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addSocre)
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
            dequeState.append([np.array(targetPositions[0]), (targetPositions[1]), (playerPositions[0]), (playerPositions[1])])
            # targetPositions, playerPositions, action, currentStopwatch, screen, timeStepforDraw = self.humanController(
            #     targetPositions, playerPositions, score, currentStopwatch, trialIndex, timeStepforDraw, dequeState,
            #     sheepNums)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pause = True
                    pg.quit()
                elif event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
            humanAction=self.joyStickController()
            action1 = np.array(humanAction[0]) * self.wolfSpeedRatio
            action2 = np.array(humanAction[1]) * self.wolfSpeedRatio
            sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, np.array(dequeState) * 10))) / 10 for i in range(sheepNums) ]


            targetPositions=[self.stayInBoundary(np.add(targetPosition, singleAction))  for (targetPosition,singleAction)  in zip(targetPositions,sheepAction)]
            playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]

            remainningTime = max(0, self.finishTime - newStopwatch)

            screen = self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)
            pg.display.update()
            eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)

            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        addSocre = [0, 0]
        if True in eatenFlag[:2]:
            # addSocre, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward
        elif True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            hunterId = hunterFlag.index(True)
            addSocre[hunterId] = self.beanReward

        else:
            results["beanEaten"] = 0
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score = np.add(score, addSocre)
        return results, targetPositions, playerPositions, score, currentStopwatch, eatenFlag, timeStepforDraw