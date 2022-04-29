import pickle
import numpy as np
import pandas as pd


class NewtonExperimentWithResetIntention():
    def __init__(self,restImage,hasRest, trial, writer,pickleWriter, experimentValues, reset, resetIntentions, drawImage):
        self.trial = trial
        self.writer = writer
        self.pickleWriter = pickleWriter
        self.experimentValues = experimentValues
        self.reset = reset
        self.drawImage = drawImage
        self.restImage = restImage
        self.resetIntentions = resetIntentions
        self.hasRest = hasRest

    def __call__(self, trailCondtions, restTimes):
        trialIndex = 0
        score = np.array([0]*self.experimentValues["numWolves"])
        # for trialIndex in range(len(trailCondtions)):
        pickleDataList = []
        for condition in trailCondtions:
            print('trial', trialIndex + 1)
            print(condition)
            sheepNums = condition['sheepNums']
            initState = self.reset(sheepNums)
            currentStopwatch = 0
            pickleResult, result, finalState, score, totalScore, currentStopwatch, eatenFlag = self.trial(
                initState, score, currentStopwatch, trialIndex, condition)
            self.resetIntentions()
            result["sheepNums"] = sheepNums
            result["totalScore"] = str(totalScore)
            pickleResult['trialIndex'] = trialIndex
            pickleResult['Name'] = self.experimentValues["name"]
            response = self.experimentValues.copy()
            response.update(result)
            pickleDataList.append(pickleResult)
            trialIndex += 1
            self.writer(response, trialIndex)
            totalTrialNum = len(trailCondtions)
            if np.mod(trialIndex, totalTrialNum/restTimes) == 0 and self.hasRest and (trialIndex < totalTrialNum):
                self.drawImage(self.restImage)
        self.pickleWriter(pickleDataList)


class NewtonExperiment():
    def __init__(self,restImage,hasRest, trial, writer,pickleWriter, experimentValues, reset, drawImage):
        self.trial = trial
        self.writer = writer
        self.pickleWriter = pickleWriter
        self.experimentValues = experimentValues
        self.reset = reset
        self.drawImage = drawImage
        self.restImage = restImage
        self.hasRest = hasRest

    def __call__(self, finishTime, trailCondtions, restTimes):
        trialIndex = 0
        score = 0
        # for trialIndex in range(len(trailCondtions)):
        pickleDataList = []
        for condition in trailCondtions:
            # condition = trailCondtions[trialIndex]
            print('trial', trialIndex + 1)
            print(condition)
            sheepNums = condition['sheepNums']
            initState = self.reset(sheepNums)
            currentStopwatch = 0
            pickleResult, result, finalState, score, currentStopwatch, eatenFlag = self.trial(
                initState, score, finishTime, currentStopwatch, trialIndex, condition)
            result["sheepNums"] = sheepNums
            result["totalScore"] = str(score)
            pickleResult['trialIndex'] = trialIndex
            response = self.experimentValues.copy()
            response.update(result)
            pickleDataList.append(pickleResult)
            trialIndex += 1
            self.writer(response, trialIndex)
            totalTrialNum = len(trailCondtions)
            if np.mod(trialIndex, totalTrialNum/restTimes) == 0 and self.hasRest and (trialIndex < totalTrialNum):
                self.drawImage(self.restImage)
        self.pickleWriter(pickleDataList)


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