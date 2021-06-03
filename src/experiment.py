import numpy as np
import pandas as pd


class NewtonExperiment():
    def __init__(self, trial, writer, experimentValues, reset, drawImage):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.reset = reset
        # self.updateWorld = updateWorld
        self.drawImage = drawImage

    def __call__(self, finishTime, trailCondtions):

        trialIndex = 0
        score = np.array([0, 0])

        # blockResult=[]
        for conditon in trailCondtions:
            sheepNums = conditon['sheepNums']
            initState = self.reset(sheepNums)
            currentStopwatch = 0
            timeStepforDraw = 0
            print('trial', trialIndex+1)
            print('Number of sheeps:', sheepNums)
            result, finalState, score, currentStopwatch, eatenFlag, timeStepforDraw = self.trial(
                initState, score, finishTime, currentStopwatch, trialIndex, timeStepforDraw, sheepNums)
            result["sheepNums"]=sheepNums
            result["score"]=str(score)
            response = self.experimentValues.copy()
            response.update(result)

            # blockResult.append({'sheepNums': sheepNums, 'score': score, 'trialTime': result["trialTime"], 'traj': result["trajectory"]})

            # if currentStopwatch >= finishTime:
            #     break
            # targetPositions = self.updateWorld(targetPositions, playerGrid, eatenFlag)
            trialIndex += 1
            responseDF = pd.DataFrame(response, index=[trialIndex])
            self.writer(responseDF)
        print(result)
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
        print(blockResult)
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