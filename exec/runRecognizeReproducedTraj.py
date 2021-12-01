# -*- coding: utf-8 -*-
from psychopy import visual, core, event
import os
import csv
import json


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
    objectSize = 20
    showTime = 1
    moveTime = 0.1

    wPtr = visual.Window(size=[768, 768], units='pix', fullscr=False)
    introText = showText(wPtr, textHeight, 'Press ENTER to start', (0, 250))
    introText.autoDraw = True
    wPtr.flip()
    keys = []
    while 'return' not in keys:
        keys = event.getKeys()
    introText.autoDraw = False

    dirName = os.path.dirname(__file__)
    csvName = 'selectedTraj.csv'
    fileName = os.path.join(dirName, '..', 'results', csvName)
    readFun = lambda key: readListCSV(fileName, key)

    # -----sheep position pre-processing-----
    trialNum = len(readCSV(fileName, 'name'))
    for i in range(trialNum):
        print('trial', i + 1)
        sheepTrajKeyName = 'sheeps traj'
        sheepNumKeyName = 'sheepNums'
        sheepPos, sheepNum = readFun(sheepTrajKeyName), readFun(sheepNumKeyName)
        sheep1No = [p for p in range(0, len(sheepPos[i]), sheepNum[i])]
        sheep2No = [p for p in range(1, len(sheepPos[i]), sheepNum[i])]
        sheepPos1 = []
        sheepPos2 = []
        for a, b in zip(sheep1No, sheep2No):
            sheepPos1.append(sheepPos[i][a])
            sheepPos2.append(sheepPos[i][b])

        play1TrajKeyName = 'player1 traj'
        play2TrajKeyName = 'player2 traj'
        play3TrajKeyName = 'player3 traj'
        playerAllPos1, playerAllPos2, playerAllPos3 = readFun(play1TrajKeyName), readFun(play2TrajKeyName), readFun(
            play3TrajKeyName)

        expandRatio = 300
        expandFun = lambda pos: expandCoordination(pos, expandRatio)
        playerPos1, playerPos2, playerPos3, = expandFun(playerAllPos1[i]), expandFun(
            playerAllPos2[i]), expandFun(playerAllPos3[i])
        sheepPos1 = expandCoordination(sheepPos1, expandRatio)
        sheepPos2 = expandCoordination(sheepPos2, expandRatio)

        drawCircleFun = lambda pos: drawCircle(wPtr, pos[0], objectSize)

        player1Traj, player2Traj, player3Traj, sheep1Traj, sheep2Traj = drawCircleFun(playerPos1), drawCircleFun(
            playerPos2), drawCircleFun(playerPos3), drawCircleFun(sheepPos1), drawCircleFun(sheepPos2),

        setColorFun = lambda traj: traj.setFillColor('black')
        setColorFun(player1Traj)
        setColorFun(player2Traj)
        setColorFun(player3Traj)
        setColorFun(sheep1Traj)
        setColorFun(sheep2Traj)
        # player1Traj.setFillColor('blue')
        # player2Traj.setFillColor('red')
        # player3Traj.setFillColor('green')
        # sheep1Traj.setFillColor('orange')
        # sheep2Traj.setFillColor('orange')

        player1Traj.autoDraw = True
        player2Traj.autoDraw = True
        player3Traj.autoDraw = True
        sheep1Traj.autoDraw = True
        sheep2Traj.autoDraw = True
        keys = []
        for x, y, z, a, b in zip(playerPos1, playerPos2, playerPos3, sheepPos1, sheepPos2):
            for t in range(showTime):
                player1Traj.setPos(x)
                player2Traj.setPos(y)
                player3Traj.setPos(z)
                sheep1Traj.setPos(a)
                sheep2Traj.setPos(b)
                wPtr.flip()
                core.wait(moveTime)
                keys = event.getKeys()
            if keys:
                print(keys)
                break

        player1Traj.autoDraw = False
        player2Traj.autoDraw = False
        player3Traj.autoDraw = False
        sheep1Traj.autoDraw = False
        sheep2Traj.autoDraw = False
        event.waitKeys()

    wPtr.flip()
    event.waitKeys()
    wPtr.close()


if __name__ == "__main__":
    main()
