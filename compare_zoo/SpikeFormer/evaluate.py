import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import numpy as np
from DataProcess import DataLoader as dl
from Model import Loss
from PIL import Image
from Metrics.Metrics import Metrics
from Model.SpikeFormer import SpikeFormer
from utils import LoadModel

if __name__ == "__main__":

    dataPath = "/home/storage2/shechen/Spike_Sample_250x400"
    spikeRadius = 32  # half length of input spike sequence expcept for the middle frame
    spikeLen = 2 * spikeRadius + 1  # length of input spike sequence
    batchSize = 4

    reuse = True
    checkPath = "CheckPoints/best.pth"

    validContainer = dl.DataContainer(dataPath=dataPath, dataType='valid',
                                      spikeRadius=spikeRadius,batchSize=batchSize)
    validData = validContainer.GetLoader()

    metrics = Metrics()
    # model = Spk2Img(spikeRadius, frameRadius, frameStride).cuda()

    model = SpikeFormer(
        inputDim=spikeLen,
        dims=(32, 64, 160, 256),  # dimensions of each stage
        heads=(1, 2, 5, 8),  # heads of each stage
        ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
        reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
        num_layers=2,  # num layers of each stage
        decoder_dim=256,  # decoder dimension
        out_channel=1  # channel of restored image
    ).cuda()


    if reuse:
        _, _, modelDict, _ = LoadModel(checkPath, model)

    model.eval()
    with torch.no_grad():
        num = 0
        pres = []
        gts = []
        for i, (spikes, gtImg) in enumerate(validData):
            B, D, H, W = spikes.size()
            spikes = spikes.cuda()
            gtImg = gtImg.cuda()
            predImg = model(spikes)
            predImg = predImg.squeeze(1)

            predImg = predImg.clamp(min=-1., max=1.)
            predImg = predImg.detach().cpu().numpy()
            gtImg = gtImg.clamp(min=-1., max=1.)
            gtImg = gtImg.detach().cpu().numpy()

            predImg = (predImg + 1.) / 2. * 255.
            predImg = predImg.astype(np.uint8)
            predImg = predImg[:, 3:-3]

            gtImg = (gtImg + 1.) / 2. * 255.
            gtImg = gtImg.astype(np.uint8)

            pres.append(predImg)
            gts.append(gtImg)
        pres = np.concatenate(pres, axis=0)
        gts = np.concatenate(gts, axis=0)

        psnr = metrics.Cal_PSNR(pres, gts)
        ssim = metrics.Cal_SSIM(pres, gts)
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()

        B, H, W = pres.shape
        divide_line = np.zeros((H, 4)).astype(np.uint8)
        for pre, gt in zip(pres, gts):
            num += 1
            concatImg = np.concatenate([pre, divide_line, gt], axis=1)
            concatImg = Image.fromarray(concatImg)
            concatImg.save('EvalResults/test_%s.jpg' % (num))

        print('*********************************************************')
        print('PSNR: %s, SSIM: %s' % (psnr, ssim))

