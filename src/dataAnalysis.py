import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def createAllCertainFormatFileList(filePath,fileFormat):
	filenameList=[os.path.join(filePath,relativeFilename) for relativeFilename in os.listdir(filePath)
		if os.path.isfile(os.path.join(filePath,relativeFilename))
		if os.path.splitext(relativeFilename)[1] in fileFormat]
	return filenameList

def cleanDataFrame(rawDataFrame):
	cleanConditionDataFrame=rawDataFrame[rawDataFrame.condition != 'None']
	cleanBeanEatenDataFrame=cleanConditionDataFrame[cleanConditionDataFrame.beanEaten!=0]
	cleanbRealConditionDataFrame=cleanBeanEatenDataFrame.loc[cleanBeanEatenDataFrame['realCondition'].isin(range(-5,6))]
	return cleanbRealConditionDataFrame

def calculateRealCondition(rawDataFrame):
	rawDataFrame['realCondition']=(np.abs(rawDataFrame['bean2GridX'] - rawDataFrame['playerGridX'])+np.abs(rawDataFrame['bean2GridY'] - rawDataFrame['playerGridY']))-(np.abs(rawDataFrame['bean1GridX'] - rawDataFrame['playerGridX'])+np.abs(rawDataFrame['bean1GridY'] - rawDataFrame['playerGridY']))
	newDataFrameWithRealCondition=rawDataFrame.copy()
	return newDataFrameWithRealCondition


def readListCSV(fileName, keyName):
	f = open(fileName, 'r')
	reader = csv.DictReader(f)
	a = []
	for i in reader:
		a.append(json.loads(i[keyName]))
	f.close()
	return a


def readCSV(fileName, keyName):
	f = open(fileName, 'r')
	reader = csv.DictReader(f)
	a = []
	for i in reader:
		a.append(i[keyName])
	f.close()
	return a


if __name__=="__main__":
	# resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Results/'
	# fileFormat = '.csv'
	# resultsFilenameList = createAllCertainFormatFileList(resultsPath, fileFormat)
	# resultsDataFrameList = [pd.read_csv(file) for file in resultsFilenameList]
	# resultsDataFrame = pd.concat(resultsDataFrameList,sort=False)
	# resultsDataFrame=calculateRealCondition(resultsDataFrame)
	# resultsDataFrame=cleanDataFrame(resultsDataFrame)
	# participantsTypeList = ['Model' if 'Model' in name else 'Human' for name in resultsDataFrame['name']]
	# resultsDataFrame['participantsType']=participantsTypeList
	# resultsDataFrame['beanEaten']=resultsDataFrame['beanEaten']-1
	# trialNumberEatNewDataFrame = resultsDataFrame.groupby(['name','realCondition','participantsType']).sum()['beanEaten']
	# trialNumberTotalEatDataFrame = resultsDataFrame.groupby(['name','realCondition','participantsType']).count()['beanEaten']
	# mergeConditionDataFrame = pd.DataFrame(trialNumberEatNewDataFrame.values/trialNumberTotalEatDataFrame.values,index=trialNumberTotalEatDataFrame.index,columns=['eatNewPercentage'])
	# mergeConditionDataFrame['eatOldPercentage']=1 - mergeConditionDataFrame['eatNewPercentage']
	# mergeParticipantsDataFrame = mergeConditionDataFrame.groupby(['realCondition','participantsType']).mean()
	# drawEatOldDataFrame=mergeParticipantsDataFrame['eatOldPercentage'].unstack('participantsType')
	# ax=drawEatOldDataFrame.plot.bar(color=['lightsalmon', 'lightseagreen'],ylim=[0.0,1.1],width=0.8)
	# pl.xticks(rotation=0)
	# ax.set_xlabel('Distance(new - old)',fontweight='bold')
	# ax.set_ylabel('Percentage of Eat Old',fontweight='bold')
	# plt.show()

	dirName = os.path.dirname(__file__)
	csvName = 'score.csv'
	fileName = os.path.join(dirName, '..', 'results', csvName)

	sheepNumKey = 'sheepNums'
	sheepConcernKey = 'sheepConcern'
	trialScoreKey = 'trialScore'

	readFun = lambda key: readListCSV(fileName, key)
	sheepNum, trialScore = readFun(sheepNumKey), readFun(trialScoreKey)
	sheepConcern = readCSV(fileName, sheepConcernKey)

	datas = {
		'sheepNum': sheepNum,
		'sheepConcern': sheepConcern,
		'trialScore': trialScore
	}
	dfTrialData = pd.DataFrame(datas)
	groupNumAndConcern = dfTrialData.groupby(['sheepConcern', 'sheepNum'])
	dfTotalScore = groupNumAndConcern.sum()  # total score for every condition
	dfAverageScore = groupNumAndConcern.mean()  # average score for every condition
	print(dfTotalScore)
	for i in range(len(dfTrialData)):
		if dfTrialData.iloc[i]["trialScore"] > 50:
			print(dfTrialData[i:i+1])

	sns.set_style("whitegrid")		# darkgrid(Default), whitegrid, dark, white, ticks
	f, ax = plt.subplots(figsize=(5, 5))
	# sns.barplot(x='sheepNum', y='trialScore', data=dfTrialData, estimator=np.mean)  # for NO sheepCorncern condition
	# barplot: Default: np.mean
	sns.barplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData, estimator=np.mean)
	# sns.boxplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData)

	# 设置坐标轴下标的字体大小
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)

	# 设置坐标名字与字体大小
	plt.ylabel('score', fontsize=10)

	# 设置X轴的各列下标字体是水平的
	plt.xticks(rotation='horizontal')

	plt.show()
