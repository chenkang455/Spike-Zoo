import os
import torch
# from torchvision import transforms
from torch.utils import data
import numpy as np
from PIL import Image
import cv2
import random


from DataProcess.DataExtactor import DataExtractor
from DataProcess.LoadSpike import LoadSpike, load_spike_raw

class Dataset(data.Dataset):

    def __init__(self, pathList, dataType, spikeRadius):

        self.pathList = pathList
        self.dataType = dataType
        self.spikeRadius = spikeRadius

        #Random Rotation
        if self.dataType == "train":
            self.choice = [0, 1, 2, 3]
        else:
            self.choice = [0]

    def __getitem__(self, index):

        spSeq, gtFrames = self.GetItem(index)

        return spSeq, gtFrames

    def __len__(self):

        return len(self.pathList)

    def GetItem(self, index):

        path = self.pathList[index]
        spSeq, gtFrames = LoadSpike(path)

        spLen, _, _ = spSeq.shape
        gtLen, _, _ = gtFrames.shape
        spCenter = spLen // 2
        gtCenter = gtLen // 2

        spLeft, spRight = (spCenter - self.spikeRadius,
                           spCenter + self.spikeRadius)
        spRight = spRight + 1
        spSeq = spSeq[spLeft:spRight]

        gtFrame = gtFrames[gtCenter]

        spSeq = np.pad(spSeq, ((0, 0), (3, 3), (0, 0)), mode='constant')
        spSeq = spSeq.astype(float) * 2 - 1

        gtFrame = gtFrame.astype(float) / 255. * 2.0 - 1.


        spSeq = torch.FloatTensor(spSeq)
        gtFrame = torch.FloatTensor(gtFrame)

        '''
        Rotate the spike frame and Gt frame by ramdom degree, 
        depending on the values of 'self.choice'
        '''
        # choice = random.choice(self.choice)
        # spSeq = torch.rot90(spSeq, choice, dims=(1,2))
        # gtFrame =torch.rot90(gtFrame, choice, dims=(1,2))
        return spSeq, gtFrame





class DataContainer():

    def __init__(self, dataPath='', dataType='train',
                 spikeRadius=16, batchSize=128, numWorks=0):

        self.dataPath = dataPath
        self.dataType = dataType
        self.spikeRadius = spikeRadius
        self.batchSize = batchSize
        self.numWorks = numWorks

        self.__GetData()

    def __GetData(self):

        dataset = None

        dataset = DataExtractor(dataPath=self.dataPath, type=self.dataType)
        self.pathList = dataset.GetData()

    def GetLoader(self):

        dataset = Dataset(self.pathList, self.dataType, self.spikeRadius)
        dataLoader = None
        if self.dataType == "train":
            dataLoader = data.DataLoader(dataset, batch_size=self.batchSize, shuffle=True,
                                         num_workers=self.numWorks, pin_memory=False)
        else:
            dataLoader = data.DataLoader(dataset, batch_size=self.batchSize, shuffle=False,
                                         num_workers=self.numWorks, pin_memory=False)

        return dataLoader

if __name__ == "__main__":

    pass



