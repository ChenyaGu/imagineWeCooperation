import pandas as pd
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

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


if __name__ == "__main__":
	dirName = os.path.dirname(__file__)
	csvName = 'Zhaolei zhuminjing niusheng.csv'
	fileName = os.path.join(dirName, '..', 'results', 'rawResults', csvName)

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
	# dfTrialData = dfTrialData[dfTrialData.sheepConcern == 'self']

	totalScore = int(dfTrialData[["trialScore"]].sum())
	groupNumAndConcern = dfTrialData.groupby(['sheepConcern', 'sheepNum'])
	dfTotalScore = groupNumAndConcern.sum()  # total score for every condition
	dfAverageScore = groupNumAndConcern.mean()  # average score for every condition
	print(dfTotalScore)

	abnormalTrial = 0
	abnormalScore = 0
	for i in range(len(dfTrialData)):
		if dfTrialData.iloc[i]["trialScore"] > 50:
			abnormalTrial += 1
			abnormalScore += dfTrialData.iloc[i]["trialScore"]
			print(dfTrialData[i:i + 1])
	print('+_+ abnormal trial number: ', abnormalTrial)
	print('+_+ abnormal trial total score: ', abnormalScore)
	print('^_^ total score: ', totalScore)
	money = float((totalScore - abnormalScore + 50 * abnormalTrial) / 100 + 20)
	print('^_^ you can get: ¥', money)

	sns.set_style("whitegrid")		# darkgrid(Default), whitegrid, dark, white, ticks
	f, ax = plt.subplots(figsize=(5, 5))
	# sns.barplot(x='sheepNum', y='trialScore', data=dfTrialData, estimator=np.mean)  # for NO sheepCorncern condition
	# barplot: Default: np.mean
	sns.barplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData, estimator=np.sum, ci=95, capsize=.05, errwidth=2, palette='Greys')
	# sns.boxplot(x='sheepConcern', y='trialScore', hue='sheepNum', data=dfTrialData)

	# 设置坐标轴下标的字体大小
	plt.xticks(fontsize=10)
	plt.yticks(fontsize=10)

	# 设置坐标名字与字体大小
	plt.ylabel('score', fontsize=10)

	# 设置X轴的各列下标字体是水平的
	plt.xticks(rotation='horizontal')

	plt.show()
