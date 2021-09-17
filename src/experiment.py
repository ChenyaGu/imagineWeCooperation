import numpy as np
import pandas as pd


class NewtonExperiment():
    def __init__(self,restImage,hasRest, trial, writer, experimentValues, reset, drawImage):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.reset = reset
        self.drawImage = drawImage
        self.restImage = restImage
        self.hasRest = hasRest
    def __call__(self, finishTime, trailCondtions,restDuration):

        trialIndex = 0
        score = np.array([0, 0])
        # for trialIndex in range(len(trailCondtions)):
        for condition in trailCondtions:
            # condition = trailCondtions[trialIndex]
            print(condition)
            sheepNums = condition['sheepNums']
            initState = self.reset(sheepNums)
            currentStopwatch = 0
            timeStepforDraw = 0
            print('trial', trialIndex+1)
            print('Number of sheeps:', sheepNums)
            result, finalState, score, playerScore1, playerScore2, totalScore, currentStopwatch, eatenFlag, timeStepforDraw = self.trial(
                initState, score, finishTime, currentStopwatch, trialIndex, timeStepforDraw, condition)
            result["sheepNums"] = sheepNums
            result["player1Score"] = str(playerScore1)
            result["player2Score"] = str(playerScore2)
            result["totalScore"] = str(totalScore)
            response = self.experimentValues.copy()
            response.update(result)
            trialIndex += 1
            self.writer(response, trialIndex)
            if np.mod(trialIndex, restDuration) == 0 and self.hasRest and (trialIndex < restDuration*4):
                # self.darwBackground()
                self.drawImage(self.restImage)
        # return result

class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath

    def __call__(self, finishTime, trailCondtions):

        trialIndex = 0
        score = np.array([0, 0])

        trialNum = 4
        blockResult=[]
        for conditon in trailCondtions:
            sheepNums = conditon['sheepNums']
            targetPositions, playerGrid = self.initialWorld(sheepNums)
            currentStopwatch = 0
            timeStepforDraw = 0
            print('trialIndex', trialIndex)
            # response = self.experimentValues.copy()
            traj, targetPositions, playerGrid, score, currentStopwatch, eatenFlag, timeStepforDraw = self.trial(
                targetPositions, playerGrid, score, currentStopwatch, trialIndex, timeStepforDraw, sheepNums)
            # response.update(results)

            blockResult.append({'sheepNums': sheepNums, 'score': score, 'traj': traj })

            if currentStopwatch >= finishTime:
                break
            targetPositions = self.updateWorld(targetPositions, playerGrid, eatenFlag)
            trialIndex += 1
        self.writer(blockResult,self.resultsPath)
        return blockResult


        
class ExperimentServer():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, resultsPath):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.resultsPath = resultsPath

    def __call__(self, finishTime, trailCondtions):

        trialIndex = 0
        score = np.array([0, 0])

        trialNum = 4
        blockResult=[]
        for conditon in trailCondtions:
            sheepNums = conditon['sheepNums']
            targetPositions, playerGrid = self.initialWorld(sheepNums)
            currentStopwatch = 0
            timeStepforDraw = 0
            print('trialIndex', trialIndex)

            traj, targetPositions, playerGrid, score, currentStopwatch, eatenFlag, timeStepforDraw = self.trial(
                targetPositions, playerGrid, score, currentStopwatch, trialIndex, timeStepforDraw, sheepNums)

            blockResult.append({'sheepNums': sheepNums, 'score': score, 'traj': traj })

            if currentStopwatch >= finishTime:
                break
            targetPositions = self.updateWorld(targetPositions, playerGrid, eatenFlag)
            trialIndex += 1
        self.writer(blockResult,self.resultsPath)
        print(blockResult)
        return blockResult