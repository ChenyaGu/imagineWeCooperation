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
from threading import Thread
from requests import get

class ThreadedClient:
    def __init__(self,initialWorld,stayInBoundary,sheepPolicy,chooseGreedyAction):
        self.initialWorld=initialWorld
        self.stayInBoundary = stayInBoundary
        self.sheepPolicy = sheepPolicy
        self.chooseGreedyAction = chooseGreedyAction
    def __call__(self, client):
        global position,targetPositions,playerPositions
        action = ''
        sheepNums=2
        # sheepNums = conditon['sheepNums']
        targetPositions, playerPositions = self.initialWorld(sheepNums)
        currentStopwatch = 0
        timeStepforDraw = 0
        while True:
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

                    # print(action)
                    position = 0
                    position=( targetPositions, playerPositions)
                    client.send(pickle.dumps(position))

                    sheepAction = [np.array(self.chooseGreedyAction(self.sheepPolicy(i, playerPositions))) / 10
                                   for i in range(sheepNums)]
                    targetPositions = [np.around(self.stayInBoundary(np.add(targetPosition, singleAction)),2) for
                                       (targetPosition, singleAction) in zip(targetPositions, sheepAction)]
                    playerPositions = [np.around(self.stayInBoundary(np.add(playerPosition, action)),2) for playerPosition, action in
                                       zip(playerPositions, [action1, action2])]
            except:
                break



def main():
    sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))

    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = 12345
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('', port))
        server.listen(5)

        ip = get('https://api.ipify.org').text
        print(f"My public IP address is: {ip}")
        print("Server is listening")
        position = ["", "", "", "", "", ["",""]]

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



        threadedClient=ThreadedClient(initialWorld,stayInBoundary,sheepPolicy,chooseGreedyAction)
        while True:
            client, address = server.accept()
            print(address)
            print(client)
            print(f"Socket Up and running with a connection from {address}")
            Thread(target = threadedClient, args = (client,)).start()
            print("Awaiting new connection")

    except KeyboardInterrupt:
        print("Closing Connection and freeing the port.")
        client.close()
        sys.exit()


if __name__ == "__main__":
    main()