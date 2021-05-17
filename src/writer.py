import pandas as pd
import os
import pickle


class WriteDataFrameToCSV():
    def __init__(self, saveResultFile):
        self.saveResultFile = saveResultFile

    def __call__(self, response, trialIndex):
        responseDF = pd.DataFrame(response, index=[trialIndex])
        if not os.path.isfile(self.saveResultFile):
            responseDF.to_csv(self.saveResultFile, header=list(responseDF.columns))
        else:
            responseDF.to_csv(self.saveResultFile, mode='a', header=False)


def loadFromPickle(path):
    pickleIn = open(path, 'rb')
    object = pickle.load(pickleIn)
    pickleIn.close()
    return object


def saveToPickle(data, path):
    pklFile = open(path, "wb")
    pickle.dump(data, pklFile)
    pklFile.close()
