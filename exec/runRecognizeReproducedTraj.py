# -*- coding: utf-8 -*-
from psychopy import visual, core, event
import os
import csv
import json
import numpy as np

def drawCircle(wPtr, pos, size):
    circle = visual.Circle(win=wPtr, lineColor='grey', pos=pos, size=size)
    return circle


def showText(wPtr, textHeight, text, position):
    introText = visual.TextStim(wPtr, height=textHeight, font='Times New Roman', text=text, pos=position)
    return introText


def expandCoordination(position, expandRatio):
    a = len(position[0])
    newPos = []
    for i in position:
        newPos.append([i[j] * expandRatio for j in range(a)])
    return newPos


def readCSV(fileName, keyName):
    f = open(fileName, 'r')
    reader = csv.DictReader(f)
    a = []
    for i in reader:
        a.append(i[keyName])
    f.close()
    return a


def readListCSV(fileName, keyName):
    f = open(fileName, 'r')
    reader = csv.DictReader(f)
    a = []
    for i in reader:
        a.append(json.loads(i[keyName]))
    f.close()
    return a


def main():
    textHeight = 24
    objectSize = 30
    targetSize = 20
    # waitTime = 0.08  # for model demo
    waitTime = 0.03  # for human demo

    wPtr = visual.Window(size=[768, 768], units='pix', fullscr=False)
    myMouse = event.Mouse(visible=True, win=wPtr)
    introText = showText(wPtr, textHeight, 'Press ENTER to start', (0, 250))
    introText.autoDraw = True
    wPtr.flip()
    keys = []
    while 'return' not in keys:
        keys = event.getKeys()
    introText.autoDraw = False
    timerText = showText(wPtr, 50, u'', (0, 0))  # 每个trial间的注视点

    dirName = os.path.dirname(__file__)
    csvName = 'DLD4s.csv'
    fileName = os.path.join(dirName, '..', 'results', 'drawTraj', csvName)
    readFun = lambda key: readListCSV(fileName, key)

    trialNum = len(readCSV(fileName, 'name'))
    startTrial = 9
    targetChoiceList = []
    stepCountList = []
    # -----target position pre-processing-----
    for i in range(startTrial, trialNum):
        print('trial', i + 1)
        targetTrajKeyName = 'sheeps traj'
        targetNumKeyName = 'sheepNums'
        targetPos, targetNum = readFun(targetTrajKeyName), readFun(targetNumKeyName)

        target1No = [p for p in range(0, len(targetPos[i]), targetNum[i])]
        targetPos1 = []
        for a in target1No:
            targetPos1.append(targetPos[i][a])
        if targetNum[i] == 2:
            target2No = [p for p in range(1, len(targetPos[i]), targetNum[i])]
            targetPos2 = []
            for b in target2No:
                targetPos2.append(targetPos[i][b])
        if targetNum[i] == 4:
            target2No = [p for p in range(1, len(targetPos[i]), targetNum[i])]
            target3No = [p for p in range(2, len(targetPos[i]), targetNum[i])]
            target4No = [p for p in range(3, len(targetPos[i]), targetNum[i])]
            targetPos2 = []
            targetPos3 = []
            targetPos4 = []
            for b, c, d in zip(target2No, target3No, target4No):
                targetPos2.append(targetPos[i][b])
                targetPos3.append(targetPos[i][c])
                targetPos4.append(targetPos[i][d])

        # -----player position pre-processing-----
        play1TrajKeyName = 'player1 traj'
        play2TrajKeyName = 'player2 traj'
        play3TrajKeyName = 'player3 traj'
        playerAllPos1, playerAllPos2, playerAllPos3 = readFun(play1TrajKeyName), readFun(play2TrajKeyName), readFun(
            play3TrajKeyName)
        expandRatio = 300
        expandFun = lambda pos: expandCoordination(pos, expandRatio)
        playerPos1, playerPos2, playerPos3, = expandFun(playerAllPos1[i]), expandFun(
            playerAllPos2[i]), expandFun(playerAllPos3[i])

        targetPos1 = expandCoordination(targetPos1, expandRatio)
        drawCircleFun = lambda pos: drawCircle(wPtr, pos[0], objectSize)
        drawTargetCircleFun = lambda pos: drawCircle(wPtr, pos[0], targetSize)
        player1Traj, player2Traj, player3Traj = drawCircleFun(playerPos1), drawCircleFun(
            playerPos2), drawCircleFun(playerPos3)
        target1Traj = drawTargetCircleFun(targetPos1)
        if targetNum[i] == 2:
            targetPos2 = expandCoordination(targetPos2, expandRatio)
            target2Traj = drawTargetCircleFun(targetPos2)
        if targetNum[i] == 4:
            targetPos2 = expandCoordination(targetPos2, expandRatio)
            targetPos3 = expandCoordination(targetPos3, expandRatio)
            targetPos4 = expandCoordination(targetPos4, expandRatio)
            target2Traj, target3Traj, target4Traj = drawTargetCircleFun(targetPos2), drawTargetCircleFun(targetPos3), drawTargetCircleFun(targetPos4)

        # only color the players
        setColorFun = lambda traj: traj.setFillColor('white')
        # setColorFun(player1Traj)
        # setColorFun(player2Traj)
        # setColorFun(player3Traj)
        # setColorFun(target1Traj)
        # if targetNum[i] == 2:
        #     setColorFun(target2Traj)
        # if targetNum[i] == 4:
        #     setColorFun(target2Traj)
        #     setColorFun(target3Traj)
        #     setColorFun(target4Traj)

        player1Traj.setFillColor('red')
        player2Traj.setFillColor('blue')
        player3Traj.setFillColor('green')

        # color all the objects for demo
        target1Traj.setFillColor('orange')
        if targetNum[i] == 2:
            target2Traj.setFillColor('DarkOrange')
        if targetNum[i] == 4:
            target2Traj.setFillColor('DarkOrange')
            target3Traj.setFillColor('SandyBrown')
            target4Traj.setFillColor('goldenrod')
        # 'orange', (255, 165, 0); 'chocolate1', (255, 127, 36); 'tan1', (255, 165, 79); 'goldenrod1', (255, 193, 37)

        player1Traj.autoDraw = True
        player2Traj.autoDraw = True
        player3Traj.autoDraw = True
        target1Traj.autoDraw = True
        if targetNum[i] == 2:
            target2Traj.autoDraw = True
        if targetNum[i] == 4:
            target2Traj.autoDraw = True
            target3Traj.autoDraw = True
            target4Traj.autoDraw = True

        stepCount = 0
        if targetNum[i] == 1:
            for x, y, z, a in zip(playerPos1, playerPos2, playerPos3, targetPos1):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    # print('press:', keys)
                    break

        if targetNum[i] == 2:
            for x, y, z, a, b in zip(playerPos1, playerPos2, playerPos3, targetPos1, targetPos2):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                target2Traj.setPos(b)
                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    while True:
                        if myMouse.isPressedIn(target1Traj):
                            choice = 'target1'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target2Traj):
                            choice = 'target2'
                            targetChoiceList.append(choice)
                            break
                    break

        if targetNum[i] == 4:
            for x, y, z, a, b, c, d in zip(playerPos1, playerPos2, playerPos3, targetPos1, targetPos2, targetPos3, targetPos4):
                stepCount += 1
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                target1Traj.setPos(a)
                target2Traj.setPos(b)
                target3Traj.setPos(c)
                target4Traj.setPos(d)
                wPtr.flip()
                core.wait(waitTime)
                keys = event.getKeys()
                if keys:
                    while True:
                        if myMouse.isPressedIn(target1Traj):
                            choice = 'target1'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target2Traj):
                            choice = 'target2'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target3Traj):
                            choice = 'target3'
                            targetChoiceList.append(choice)
                            break
                        if myMouse.isPressedIn(target4Traj):
                            choice = 'target4'
                            targetChoiceList.append(choice)
                            break
                    break

        # print('choice:', choice)
        print('stop step:', stepCount)
        stepCountList.append(stepCount)
        player1Traj.autoDraw = False
        player2Traj.autoDraw = False
        player3Traj.autoDraw = False
        target1Traj.autoDraw = False
        if targetNum[i] == 2:
            target2Traj.autoDraw = False
        if targetNum[i] == 4:
            target2Traj.autoDraw = False
            target3Traj.autoDraw = False
            target4Traj.autoDraw = False
        event.waitKeys()
        '''
        # Put fixation points or rests
        restTime = 2
        restDuration = (trialNum-startTrial)/restTime
        if np.mod((i-startTrial+1), restDuration) != 0:
            dtimer = core.CountdownTimer(1)  # wait for 1s
            while dtimer.getTime() > 0:
                timerText.text = '+'
                timerText.bold = True
                timerText.draw()
            wPtr.flip()
        else:   # rest
            restText = showText(wPtr, textHeight, 'Press Space to continue', (0, 300))
            restText.autoDraw = True
            wPtr.flip()
            event.waitKeys()
            restText.autoDraw = False
        '''

    wPtr.flip()
    event.waitKeys()
    wPtr.close()


if __name__ == "__main__":
    main()
