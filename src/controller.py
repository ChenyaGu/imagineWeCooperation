import numpy as np
import pygame as pg
import random
import itertools as it
import math
import os
import collections as co
class JoyStickForceControllers():

    def __init__(self):
        pg.joystick.init()
        joyStickIdList = range(pg.joystick.get_count())
        self.joystickList = [pg.joystick.Joystick(joyStickId) for joyStickId in joyStickIdList ]
        [joystick.init() for joystick in self.joystickList]

    def __call__(self):
        actionList = []
        [joystick.init() for joystick in self.joystickList]
        for joystick in self.joystickList:
            action = [0, 0, 0, 0, 0]
            numAxes = joystick.get_numaxes()


            for i in range(2):
                # axis = joystick.get_axis(i)
                if abs(joystick.get_axis(i)) > 0.5:
                    action[2*i+1] =joystick.get_axis(i)/abs(joystick.get_axis(i))*(abs(joystick.get_axis(i)) - 0.5) *2
               
                
                    # print(axis)
            actionList.append(action)

        # joystickSpaceSize = len(self.joystickList) * numAxes

        # actionList = [0 if abs(actionList[i]) < 0.5 else actionList[i] for i in range(joystickSpaceSize)]
        # action = [actionList[i:i + 2] for i in range(0, len(actionList), numAxes)]
        return actionList

class JoyStickController():

    def __init__(self,joyStickId):
        pg.joystick.init()
        
        self.joystick = pg.joystick.Joystick(joyStickId)
        self.joystick.init()

    def __call__(self):
        actionList=[]
        self.joystick.init()
        numAxes = self.joystick.get_numaxes()

        for event in pg.event.get():  # User did something
            if event.type == pg.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

            # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
            if event.type == pg.JOYBUTTONDOWN:
                print("Joystick button pressed.")
            if event.type == pg.JOYBUTTONUP:
                print("Joystick button released.")
        for i in range(numAxes):
            # axis = joystick.get_axis(i)
            if abs( self.joystick.get_axis(i)) > 0.5:
                sign =  self.joystick.get_axis(i) / abs( self.joystick.get_axis(i))
                axis = sign * math.log(9 * abs( self.joystick.get_axis(i)) + 1) / 2.303
            else:
                axis =  self.joystick.get_axis(i)
                # print(axis)
            actionList.append(axis)

        joystickSpaceSize = numAxes

        actionList = [0 if abs(actionList[i]) < 0.5 else actionList[i] for i in range(joystickSpaceSize)]
        action = actionList[0:2]
        print(action)
        return  action

class JoyStickControllers():

    def __init__(self,joyStickIdList):
        pg.joystick.init()
        joyStickIdList = range(pg.joystick.get_count())
        self.joystickList = [pg.joystick.Joystick(joyStickId) for joyStickId in joyStickIdList ]
        [joystick.init() for joystick in self.joystickList]

    def __call__(self):
        actionList = []
        [joystick.init() for joystick in self.joystickList]
        for joystick in self.joystickList:
            numAxes = joystick.get_numaxes()
            for i in range(numAxes):
                # axis = joystick.get_axis(i)
                if abs(joystick.get_axis(i)) > 0.5:
                    sign = joystick.get_axis(i) / abs(joystick.get_axis(i))
                    axis = sign * math.log(9 * abs(joystick.get_axis(i)) + 1) / 2.303
                else:
                    axis = joystick.get_axis(i)
                    # print(axis)
                actionList.append(axis)

        joystickSpaceSize = len(self.joystickList) * numAxes

        actionList = [0 if abs(actionList[i]) < 0.5 else actionList[i] for i in range(joystickSpaceSize)]
        action = [actionList[i:i + 2] for i in range(0, len(actionList), numAxes)]
        print(action)
        return  action



class HumanController():
    def __init__(self, writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, sheepPolicy, chooseGreedyAction):
        self.writer = writer
        self.gridSize = gridSize
        self.stopwatchEvent = stopwatchEvent
        self.stopwatchUnit = stopwatchUnit
        # self.stopwatch = 0
        self.wolfSpeedRatio = wolfSpeedRatio
        self.finishTime = finishTime
        self.drawNewState = drawNewState
        self.stayInBoundary = stayInBoundary
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir
        self.sheepPolicy = sheepPolicy
        self.chooseGreedyAction = chooseGreedyAction

    def __call__(self, targetPositions, playerPositions, currentScore, currentStopwatch, trialIndex, timeStepforDraw, dequeState,sheepNums):
        newStopwatch = currentStopwatch
        remainningTime = max(0, self.finishTime - currentStopwatch)

        screen = self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)


        # self.writer(results, timeStepforDraw)

        if self.saveImage == True:
            if not os.path.exists(self.saveImageDir):
                os.makedirs(self.saveImageDir)
            pg.image.save(screen, self.saveImageDir + '/' + format(timeStepforDraw, '04') + ".png")
        timeStepforDraw += 1

        pg.joystick.init()
        joystickCount = pg.joystick.get_count()

        pause = True

        action1 = [0, 0]
        action2 = [0, 0]
        action3 = [0, 0]
        action4 = [0, 0]

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pause = True
                pg.quit()
            elif event.type == self.stopwatchEvent:
                newStopwatch = newStopwatch + self.stopwatchUnit

        actionList = []
        for i in range(joystickCount):
            joystick = pg.joystick.Joystick(i)
            joystick.init()
            numAxes = joystick.get_numaxes()

            for i in range(numAxes):
                # axis = joystick.get_axis(i)
                if abs(joystick.get_axis(i)) > 0.5:
                    sign = joystick.get_axis(i) / abs(joystick.get_axis(i))
                    axis = sign * math.log(9 * abs(joystick.get_axis(i)) + 1) / 2.303
                else:
                    axis = joystick.get_axis(i)


                actionList.append(axis)
                # print(actionList)

        joystickSpaceSize = joystickCount * numAxes
        actionList = [0 if abs(actionList[i]) < 0.5 else actionList[i] for i in range(joystickSpaceSize)]
        action = [actionList[i:i + 2] for i in range(0, len(actionList), numAxes)]

        action1 = np.array(action[0]) * self.wolfSpeedRatio
        action2 = np.array(action[1]) * self.wolfSpeedRatio
        print(action1,action2)


        sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, np.array(dequeState) * 10))) / 10 for i in range(sheepNums) ]


        targetPositions=[self.stayInBoundary(np.add(targetPosition, singleAction))  for (targetPosition,singleAction)  in zip(targetPositions,sheepAction)]
        playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]

        remainningTime = max(0, self.finishTime - newStopwatch)

        screen = self.drawNewState(targetPositions, playerPositions, remainningTime, currentScore)
        pg.display.update()
        return targetPositions, playerPositions, action, newStopwatch, screen, timeStepforDraw


def calculateSoftmaxProbability(probabilityList, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, probabilityList)), np.sum(np.exp(np.multiply(beta, probabilityList)))))
    return newProbabilityList


class ModelController():
    def __init__(self, policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita):
        self.policy = policy
        self.gridSize = gridSize
        self.stopwatchEvent = stopwatchEvent
        self.stopwatchUnit = stopwatchUnit
        self.stopwatch = 0
        self.drawNewState = drawNewState
        self.finishTime = finishTime
        self.softmaxBeita = softmaxBeita

    def __call__(self, targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch):
        pause = True
        newStopwatch = currentStopwatch
        remainningTime = max(0, self.finishTime - currentStopwatch)
        self.drawNewState(targetPositionA, targetPositionB, playerPosition, remainningTime, currentScore)
        while pause:
            targetStates = (tuple(targetPositionA), tuple(targetPositionB))
            if targetStates not in self.policy.keys():
                targetStates = (tuple(targetPositionB), tuple(targetPositionA))
            policyForCurrentStateDict = self.policy[targetStates][tuple(playerPosition)]
            if self.softmaxBeita < 0:
                actionMaxList = [action for action in policyForCurrentStateDict.keys() if policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
                action = random.choice(actionMaxList)
            else:
                actionProbability = np.divide(list(policyForCurrentStateDict.values()), np.sum(list(policyForCurrentStateDict.values())))
                softmaxProbabilityList = calculateSoftmaxProbability(list(actionProbability), self.softmaxBeita)
                action = list(policyForCurrentStateDict.keys())[list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
            playerNextPosition = np.add(playerPosition, action)
            if np.any(playerNextPosition < 0) or np.any(playerNextPosition >= self.gridSize):
                playerNextPosition = playerPosition
            pause = False
            for event in pg.event.get():
                if event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
                    remainningTime = max(0, self.finishTime - newStopwatch)
            self.drawNewState(targetPositionA, targetPositionB, playerNextPosition, remainningTime, currentScore)
            pg.display.flip()
        return playerNextPosition, action, newStopwatch


if __name__ == "__main__":
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 20
    leaveEdgeSpace = 2
    lineWidth = 2
    backgroundColor = [188, 188, 0]
    lineColor = [255, 255, 255]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    targetPositionA = [5, 5]
    targetPositionB = [15, 5]
    playerPosition = [10, 15]
    currentScore = 5
    textColorTuple = (255, 50, 50)
    stopwatchEvent = pg.USEREVENT + 1
    stopwatchUnit = 10
    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    finishTime = 90000
    currentStopwatch = 32000
    softmaxBeita = 20

    drawBackground = Visualization.DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = Visualization.DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

    getHumanAction = HumanController(gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)

    import pickle
    policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl", "rb"))
    getModelAction = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita)


    [playerNextPosition, action, newStopwatch] = getModelAction(targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch)
    print(playerNextPosition, action, newStopwatch)

    pg.quit()
