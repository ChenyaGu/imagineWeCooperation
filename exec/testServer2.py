# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:55:18 2020

@author: arthurtan
"""
import sys
import os
import socket
import pickle
import numpy as np
import threading
from threading import Thread

from requests import get
from src.writer import saveToPickle
import time
class ThreadedClient:
    def __init__(self,conditions,initialWorld,stayInBoundary,sheepPolicy,chooseGreedyAction,checkEaten,checkTerminationOfTrial,writer):
        self.conditions=conditions
        self.initialWorld=initialWorld
        self.stayInBoundary = stayInBoundary
        self.sheepPolicy = sheepPolicy
        self.chooseGreedyAction = chooseGreedyAction
        self.checkEaten=checkEaten
        self.checkTerminationOfTrial=checkTerminationOfTrial
        self.beanReward=1
        self.writer = writer
    def __call__(self, client):
        global position,targetPositions,playerPositions,conditionId,score,readyToStart,bothReady,waitReady
        action = ''


        bothReady=[0,0]
        waitReady= False
        while not waitReady:
            try:
                ready = client.recv(2048)
                if not ready:
                    client.send(str.encode("Goodbye"))
                    break
                else:
                    ready = pickle.loads(ready)
                    bothReady[ready[0]]=ready[1]

                    waitReady = bothReady[0] and bothReady[1]
                    print(bothReady,waitReady)
                    client.send(pickle.dumps(waitReady))
            except:
                print('break')

                break

        # sheepNums=2
        # sheepNums = conditon['sheepNums']
        # targetPositions, playerPositions = self.initialWorld(sheepNums)
        currentStopwatch = 0

        timeStepforDraw = 0
        score=[0,0]
        blockResult = []
        conditionId=0
        print(time.time(),threading.current_thread().name)
        while conditionId<len(self.conditions):
        # for conditionId in range(len(self.conditions)):

            condition=self.conditions[conditionId]


            sheepNums = condition['sheepNums']
            print(conditionId,sheepNums,threading.current_thread().name )
            targetPositions, playerPositions = self.initialWorld(sheepNums)
            pause=True
            traj=[]
            while pause:
                try:

                    action1=[0,0]
                    action2=[0,0]
                    data = client.recv(2048)

                    if not data:
                        client.send(str.encode("Goodbye"))
                        break
                    else:
                        action = pickle.loads(data)
                        if action[0] == 0:
                            action1 = action[1]
                        elif action[0] == 1:
                            action2 = action[1]
                        # print(action[0])
                        # print(action)
                        position = 0
                        position=( list(targetPositions), list(playerPositions),list(score))
                        client.send(pickle.dumps(position))

                        sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, playerPositions))) / 10
                                       for i in range(sheepNums)]

                        state={'playerPositions':playerPositions,'targetPositions':targetPositions,'agentId':action[0],'action':action,'sheepAction':sheepAction}

                        targetPositions = [np.around(self.stayInBoundary(np.add(targetPosition, singleAction)),2) for
                                           (targetPosition, singleAction) in zip(targetPositions, sheepAction)]
                        playerPositions = [np.around(self.stayInBoundary(np.add(playerPosition, action)),2) for playerPosition, action in
                                           zip(playerPositions, [action1, action2])]

                        eatenFlag, hunterFlag = self.checkEaten(targetPositions, playerPositions)
                        pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
                        addSocre = [0, 0]
                        if True in eatenFlag[:2]:
                            # addSocre, timeStepforDraw = self.attributionTrail(eatenFlag, hunterFlag, timeStepforDraw)
                            # results["beanEaten"] = eatenFlag.index(True) + 1
                            hunterId = hunterFlag.index(True)
                            addSocre[hunterId] = self.beanReward
                        elif True in eatenFlag:
                            # results["beanEaten"] = eatenFlag.index(True) + 1
                            hunterId = hunterFlag.index(True)
                            addSocre[hunterId] = self.beanReward

                        score = np.add(score, addSocre)
                        state.update({'playerPositionsAfterAction':playerPositions,'targetPositionsAfterAction':targetPositions,'eatenFlag':eatenFlag,'hunterFlag':hunterFlag})

                        traj.append(state)

                except:
                    break
                blockResult.append({'sheepNums': sheepNums, 'score': score, 'traj': traj})
            conditionId = conditionId + 1

        dirName = os.path.dirname(__file__)
        resultsDicPath = os.path.join(dirName, '..', 'results')
        writerPath = os.path.join(resultsDicPath, str(action[0])) + '.pickle'


        self.writer(blockResult, writerPath)



def main():
    sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = 12345
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('', port))
        server.listen(5)
        ip='192.168.1.1'
        # ip = get('https://api.ipify.org').text
        print(f"My public IP address is: {ip}")
        print("Server is listening")
        position = ["", "", "", "", "", ["",""]]
        import itertools as it
        from collections import OrderedDict
        manipulatedVariables = OrderedDict()
        manipulatedVariables['sheepNums'] = [2, 4, 8]
        trailNumEachCondition = 3

        productedValues = it.product(
            *[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
        parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

        AllConditions = parametersAllCondtion * trailNumEachCondition

        from src.updateWorld import InitialWorld, StayInBoundary
        gridSize = 60
        bounds = [0, 0, gridSize - 1, gridSize - 1]
        minDistanceForReborn = 5
        numPlayers = 2
        initialWorld = InitialWorld(bounds, numPlayers, minDistanceForReborn)

        xBoundary = [bounds[0], bounds[2]]
        yBoundary = [bounds[1], bounds[3]]
        stayInBoundary = StayInBoundary(xBoundary, yBoundary)

        from src.sheepPolicy import RandomMovePolicy, chooseGreedyAction
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        sheepPolicy = RandomMovePolicy(sheepActionSpace)

        from src.trial import CheckEaten, CheckTerminationOfTrial,isAnyKilled
        killzone = 2
        checkEaten = CheckEaten(killzone, isAnyKilled)
        finishTime = 1000 * 15
        checkTerminationOfTrial = CheckTerminationOfTrial(finishTime)


        threadedClient=ThreadedClient(AllConditions,initialWorld,stayInBoundary,sheepPolicy,chooseGreedyAction,checkEaten,checkTerminationOfTrial,saveToPickle)



        while True:
            client, address = server.accept()
            print(address)
            print(client)
            print("Socket Up and running with a connection from {address}")
            Thread(target = threadedClient, args = (client,)).start()
            print("Awaiting new connection")

    except KeyboardInterrupt:
        print("Closing Connection and freeing the port.")
        client.close()
        sys.exit()


if __name__ == "__main__":
    main()