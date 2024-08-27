import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import numpy as np
from PIL import Image
from Metrics.Metrics import Metrics
from Model.SpikeFormer import SpikeFormer
from DataProcess.LoadSpike import load_spike_raw
from utils import LoadModel
import shutil
from PIL import Image

def PredictImg(model, inputs):
    inputs = torch.FloatTensor(inputs)
    inputs = inputs.cuda()

    predImg = model(inputs).squeeze(dim=1)

    predImg = predImg.clamp(min=-1., max=1.)
    predImg = predImg.detach().cpu().numpy()
    predImg = (predImg + 1.) / 2. * 255.
    predImg = np.clip(predImg, 0., 255.)
    predImg = predImg.astype(np.uint8)
    predImg = predImg[:, 3:-3]

    return predImg

if __name__ == "__main__":

    dataName = "reds"
    spikeRadius = 32
    spikeLen = 2 * spikeRadius + 1
    stride = 32
    batchSize = 8
    reuse = True
    checkPath = "best.pth"
    sceneClass = {
                   1:'ballon.dat', 2:'car-100kmh.dat',
                   3:'forest.dat', 4:'railway.dat',
                   5:'rotation1.dat', 6:'rotation2.dat',
                   7:'train-350kmh.dat', 8:'viaduct-bridge.dat'
    }
    sceneName = sceneClass[2]
    dataPath = "/home/storage1/Dataset/SpikeImageData/RealData/%s" %(sceneName)
    resultPath = sceneName + "_stride_" + str(stride) + "/"
    shutil.rmtree(resultPath) if os.path.exists(resultPath) else os.mkdir(resultPath)
    spikes = load_spike_raw(dataPath)
    totalLen = spikes.shape[0]
    metrics = Metrics()
    model = SpikeFormer(
        inputDim=spikeLen,
        dims=(32, 64, 160, 256),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=2,  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        out_channel = 1  # channel of restored image
    ).cuda()

    if reuse:
        _, _, modelDict, _ = LoadModel(checkPath, model)

    model.eval()
    with torch.no_grad():
        num = 0
        pres = []
        batchFlag = 1
        inputs = np.zeros((batchSize, spikeLen, 256, 400))  # 65
        for i in range(32, totalLen - 32, stride):
            batchFlag = 1
            spike = spikes[i - spikeRadius: i + spikeRadius + 1]
            spike = np.pad(spike, ((0, 0), (3, 3), (0, 0)), mode='constant')
            spike = spike.astype(float) * 2 - 1
            inputs[num % batchSize] = spike
            num += 1

            if num % batchSize == 0:
                predImg = PredictImg(model, inputs)
                inputs = np.zeros((batchSize, spikeLen, 256, 400))  # 65
                pres.append(predImg)
                batchFlag = 0

        if batchFlag == 1:
            imgNum = num % batchSize
            inputs = inputs[0: imgNum]
            predImg = PredictImg(model, inputs)
            inputs = np.zeros((batchSize, spikeLen, 256, 400))
            pres.append(predImg)

        predImgs = np.concatenate(pres, axis=0)
        count = 0
        for img in predImgs:
            count += 1
            img = Image.fromarray(img)
            img.save(resultPath + '%s.jpg' % (count))

