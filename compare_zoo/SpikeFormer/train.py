import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch import optim
import numpy as np
from DataProcess import DataLoader as dl
from Model.SpikeFormer import SpikeFormer
from Metrics.Metrics import Metrics
from Model import Loss
from utils import SaveModel, LoadModel
from PIL import Image

def eval(model, validData, epoch, optimizer, metrics):

    model.eval()
    print('Eval Epoch: %s' %(epoch))

    with torch.no_grad():
        pres = []
        gts = []
        for i, (spikes, gtImg) in enumerate(validData):

            spikes = spikes.cuda()
            gtImg = gtImg.cuda()
            predImg = model(spikes)
            predImg = predImg.squeeze(1)
            predImg = predImg[:,3:-3,:]

            predImg = predImg.clamp(min=-1., max=1.)
            predImg = predImg.detach().cpu().numpy()
            gtImg = gtImg.clamp(min=-1., max=1.)
            gtImg = gtImg.detach().cpu().numpy()

            predImg = (predImg + 1.) / 2. * 255.
            predImg = predImg.astype(np.uint8)

            gtImg = (gtImg + 1.) / 2. * 255.
            gtImg = gtImg.astype(np.uint8)

            pres.append(predImg)
            gts.append(gtImg)
        pres = np.concatenate(pres, axis=0)
        gts = np.concatenate(gts, axis=0)

        psnr = metrics.Cal_PSNR(pres, gts)
        ssim = metrics.Cal_SSIM(pres, gts)
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()

        SaveModel(epoch, (psnr, ssim), model, optimizer, saveRoot)
        if psnr >= best_psnr and ssim >= best_ssim:
            metrics.Update(psnr, ssim)
            SaveModel(epoch, (psnr, ssim), model, optimizer, saveRoot, best=True)
            with open('eval_best_log.txt', 'w') as f:
                f.write('epoch: %s; psnr: %s, ssim: %s\n' %(epoch, psnr, ssim))
            B, H, W = pres.shape
            divide_line = np.zeros((H,4)).astype(np.uint8)
            num = 0
            for pre, gt in zip(pres, gts):
                num += 1
                concatImg = np.concatenate([pre, divide_line, gt], axis=1)
                concatImg = Image.fromarray(concatImg)
                concatImg.save('EvalResults/valid_%s.jpg' % (num))

        print('*********************************************************')
        best_psnr, best_ssim, _ = metrics.GetBestMetrics()
        print('Eval Epoch: %s, PSNR: %s, SSIM: %s, Best_PSNR: %s, Best_SSIM: %s'
              %(epoch, psnr, ssim, best_psnr, best_ssim))

    model.train()

def Train(trainData, validData, model, optimizer, epoch, start_epoch, metrics, saveRoot, perIter):
    avg_l2_loss = 0.
    avg_vgg_loss = 0.
    avg_edge_loss = 0.
    avg_total_loss = 0.
    l2loss = Loss.CharbonnierLoss()
    vggloss = Loss.VGGLoss4('vgg19-low-level4.pth').cuda()
    criterion_edge = Loss.EdgeLoss()
    LAMBDA_L2 = 100.0
    LAMBDA_VGG = 1.0
    LAMBDA_EDGE = 5.0   
    for i in range(start_epoch, epoch):
        for iter, (spikes, gtImg) in enumerate(trainData):
            spikes = spikes.cuda()
            gtImg = gtImg.cuda()
            predImg = model(spikes)
            gtImg = gtImg.unsqueeze(1)
            predImg = predImg[:,:,3:-3,:]

            loss_vgg = vggloss(gtImg, predImg) * LAMBDA_VGG
            loss_l2 = l2loss(gtImg, predImg) * LAMBDA_L2
            loss_edge = criterion_edge(gtImg, predImg) * LAMBDA_EDGE

            totalLoss = loss_l2 + loss_vgg + loss_edge

            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            avg_l2_loss += loss_l2.detach().cpu()
            avg_vgg_loss += loss_vgg.detach().cpu()
            avg_edge_loss += loss_edge.detach().cpu()
            avg_total_loss += totalLoss.detach().cpu()
            if (iter + 1) % perIter == 0:
                avg_l2_loss = avg_l2_loss / perIter
                avg_vgg_loss = avg_vgg_loss / perIter
                avg_edge_loss = avg_edge_loss / perIter
                avg_total_loss = avg_total_loss / perIter
                print('=============================================================')
                print('Epoch: %s, Iter: %s' % (i, iter + 1))
                print('L2Loss: %s; VggLoss: %s; EdgeLoss: %s; TotalLoss: %s' % (
                    avg_l2_loss.item(), avg_vgg_loss.item(), avg_edge_loss.item(), avg_total_loss.item()))
                avg_l2_loss = 0.
                avg_vgg_loss = 0.
                avg_edge_loss = 0.
                avg_total_loss = 0.

        if (i + 1) % 1 == 0:
            eval(model, validData, i, optimizer, metrics)

if __name__ == "__main__":

    dataPath = "/home/storage2/shechen/Spike_Sample_250x400"
    spikeRadius = 32  # half length of input spike sequence expcept for the middle frame
    spikeLen = 2 * spikeRadius + 1  # length of input spike sequence
    batchSize = 2
    epoch = 200
    start_epoch = 0
    lr = 2e-4
    saveRoot = "CheckPoints/"  # path to save the trained model
    perIter = 20

    reuse = False
    reuseType = 'latest'  # 'latest' or 'best'
    checkPath = os.path.join('CheckPoints', '%s.pth' % (reuseType))

    trainContainer = dl.DataContainer(dataPath=dataPath, dataType='train',
                                      spikeRadius=spikeRadius,
                                      batchSize=batchSize)
    trainData = trainContainer.GetLoader()

    validContainer = dl.DataContainer(dataPath=dataPath, dataType='valid',
                                      spikeRadius=spikeRadius,
                                      batchSize=batchSize)
    validData = validContainer.GetLoader()

    metrics = Metrics()

    model = SpikeFormer(
        inputDim = spikeLen,
        dims = (32, 64, 160, 256),      # dimensions of each stage
        heads = (1, 2, 5, 8),           # heads of each stage
        ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
        reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        num_layers = 2,                 # num layers of each stage
        decoder_dim = 256,              # decoder dimension
        out_channel = 1                 # channel of restored image
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=False)

    if reuse:
        preEpoch, prePerformance, modelDict, optDict = LoadModel(checkPath, model, optimizer)
        start_epoch = preEpoch + 1
        psnr, ssim = prePerformance[0], prePerformance[1]
        metrics.Update(psnr, ssim)
        for para in optimizer.param_groups:
            para['lr'] = lr

    model.train()

    Train(trainData, validData, model, optimizer, epoch, start_epoch,
                    metrics, saveRoot, perIter)
