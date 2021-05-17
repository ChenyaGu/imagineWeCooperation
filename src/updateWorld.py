import random
import numpy as np
import copy


def computeAngleBetweenTwoVectors(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    lenthOfVector1 = np.sqrt(vector1.dot(vector1))
    lenthOfVector2 = np.sqrt(vector2.dot(vector2))
    cosAngle = vector1.dot(vector2) / (lenthOfVector1 * lenthOfVector2)
    angle = np.arccos(cosAngle)
    return angle


def indexCertainNumberInList(list, number):
    indexList = [i for i in range(len(list)) if list[i] == number]
    return indexList


def samplePosition(bounds):
    positionX = np.random.uniform(bounds[0], bounds[2])
    positionY = np.random.uniform(bounds[1], bounds[3])
    position = [positionX, positionY]
    return position


class InitialWorld():
    def __init__(self, bounds, numPlayers, minDistance):
        self.bounds = bounds
        self.numPlayers = numPlayers
        self.minDistance = minDistance

    def __call__(self,numSheep):

        initPlayerGrids = [samplePosition(self.bounds) for i in range(self.numPlayers)]
        targetPositions = [samplePosition(self.bounds) for i in range(numSheep)]

        for i,targetPos in enumerate(targetPositions):
            proposalPos = targetPos
            while np.all(np.array([np.linalg.norm(np.array(humanGrid) - np.array(targetPos)) for humanGrid in initPlayerGrids]) < self.minDistance):
                proposalPos = samplePosition(self.bounds)
            targetPositions[i] = proposalPos


        return targetPositions,initPlayerGrids


class UpdateWorld():
    def __init__(self, bounds, minDistanceForReborn):
        self.bounds = bounds
        self.minDistanceForReborn = minDistanceForReborn

    def __call__(self, targetGrid, playerGrid, eatenFlag):
        catchedSheepId=eatenFlag.index(True)
        newTargetGrid = samplePosition(self.bounds)
        while np.any(np.array([np.linalg.norm(np.array(humanGrid) - np.array(newTargetGrid)) for humanGrid in playerGrid]) < self.minDistanceForReborn):
            newTargetGrid = samplePosition(self.bounds)
        targetGrid[catchedSheepId]=newTargetGrid

        return  targetGrid


class StayInBoundary:
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        adjustedX, adjustedY = position
        if position[0] >= self.xMax:
            adjustedX = self.xMax
        if position[0] <= self.xMin:
            adjustedX = self.xMin
        if position[1] >= self.yMax:
            adjustedY = self.yMax
        if position[1] <= self.yMin:
            adjustedY = self.yMin
        checkedPosition = (adjustedX, adjustedY)
        return checkedPosition


def main():
    dimension = 15
    bounds = [0, 0, dimension - 1, dimension - 1]
    condition = [-5, -3, -1, 0, 1, 3, 5]
    counter = [0] * len(condition)
    minDistanceBetweenGrids = 1
    initialWorld = InitialWorld(bounds)
    target1Grid, target2Grid, playerGrid = initialWorld(minDistanceBetweenGrids)
    updateWorld = UpdateWorld(bounds, condition, counter)
    target2Grid, nextCondition = updateWorld(target1Grid, playerGrid)
    print(playerGrid, target2Grid, nextCondition)


if __name__ == "__main__":
    main()
