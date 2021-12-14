import os
import pandas as pd
import pickle
import glob


def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def SortTrajectories():
    dirName = os.path.dirname(__file__)
    dataPath = os.path.join(dirName, '..', 'results')
    resultPath = glob.glob(os.path.join(dataPath, '*.pickle'))
    data = [loadFromPickle(resultPath[i]) for i in range(len(resultPath))]
    dfData = pd.DataFrame(data)
    print(dfData)
    return dfData


def flat(data):
    flatList = []
    for i in data:
        if isinstance(i, list):
            flatList.extend(flat(i))
        else:
            flatList.append(i)
    return flatList


def main():
    dataList = SortTrajectories()
    a = flat(dataList)
    # print(a)


if __name__ == "__main__":
    main()