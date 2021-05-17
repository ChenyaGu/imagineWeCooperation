#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:53:08 2020

@author: arthurtan
"""

import socket
import pickle
import math
import pygame as pg
class Client:
    def __init__(self, ipAddress, port, clientNumber):
        self.server = socket.socket()
        self.ipAddress = ipAddress
        self.port = port
        self.clientNumber = clientNumber
        try:
            self.server.connect((self.ipAddress, self.port))
            print("Connected to server")
        except Exception as e:
            print("something's wrong with %s:%d. Exception is %s" % (self.ipAddress, self.port, e))

    def send(self, data):
        self.server.send(pickle.dumps(data))

    def receive(self):
        data = self.server.recv(2048)
        allPositions = pickle.loads(data)
        return allPositions


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

    ipAddress = '192.168.199.212'
    port = 12345
    clientNumber = int(input("Input your client number(0 or 1): "))
    client = Client(ipAddress, port, clientNumber)
    import numpy as np
    from src.controller import JoyStickController

    control = JoyStickController(clientNumber)

    pause = True

    # pg.joystick.init()
    pg.init()
    joystickCount = pg.joystick.get_count()
    # print(joystickCount)
    pause = True


    screenWidth = 800
    screenHeight = 800
    screenCenter = [screenWidth / 2, screenHeight / 2]
    fullScreen = False
    from src.visualization import DrawBackground, DrawNewState,InitializeScreen
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
    gridSize = 60
    leaveEdgeSpace = 6
    lineWidth = 1
    from pygame.color import THECOLORS
    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [THECOLORS['blue']] * 16  # [255, 50, 50]
    playerColors = [THECOLORS['orange'], THECOLORS['red']]
    textColorTuple = THECOLORS['green']
    targetRadius = 10
    playerRadius = 10
    totalBarLength = 100
    barHeight = 20

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple, playerColors)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    action1 = [0, 0]
    action2 = [0, 0]
    action3 = [0, 0]
    action4 = [0, 0]

    # while pause:
    #     for event in pg.event.get():
    #         if event.type == pg.QUIT:
    #             pause = True
    #             pg.quit()
    #
    #
    #     actionList = []
    #     for i in range(joystickCount):
    #         joystick = pg.joystick.Joystick(i)
    #         joystick.init()
    #         numAxes = joystick.get_numaxes()
    #         print(numAxes)
    #         for i in range(numAxes):
    #             # axis = joystick.get_axis(i)
    #             if abs(joystick.get_axis(i)) > 0.5:
    #                 sign = joystick.get_axis(i) / abs(joystick.get_axis(i))
    #                 axis = sign * math.log(9 * abs(joystick.get_axis(i)) + 1) / 2.303
    #             else:
    #                 axis = joystick.get_axis(i)
    #
    #             actionList.append(axis)
    #             # print(actionList)
    #
    #     joystickSpaceSize = joystickCount * numAxes
    #     actionList = [0 if abs(actionList[i]) < 0.5 else actionList[i] for i in range(joystickSpaceSize)]
    #     action = [actionList[i:i + 2] for i in range(0, len(actionList), numAxes)]
    #
    #     action1 = np.array(action[0]) * 1
    #     action2 = np.array(action[1]) * 1
    #     # print(action1, action2)

    while pause:
        # action = np.around(control(),decimals=2)
        # action = [0,1]
        # print((clientNumber,action))
        #
        #
        # print(action)
        pg.time.delay(32)
        action=control()
        print('joyaction',action)
        client.send((clientNumber,action))



        def receiveAction():
            reply = client.receive()
            if reply == "":
                return "null"
            else:
                return reply


        position = receiveAction()
        targetPositions=position[0]
        playerPositions=position[1]
        screen = drawNewState(targetPositions, playerPositions, 0, [0,0])
        pg.display.flip()
        print('position',position)
        # print(position)