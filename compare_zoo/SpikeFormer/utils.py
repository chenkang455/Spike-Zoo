import os
import torch

def SaveModel(epoch, bestPerformance, model, optimizer, saveRoot, best=False):
    saveDict = {
        'pre_epoch':epoch,
        'performance':bestPerformance,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }
    savePath = os.path.join(saveRoot, '%s.pth' %('latest' if not best else 'best'))
    torch.save(saveDict, savePath)

def LoadModel(checkPath, model, optimizer=None):
    stateDict = torch.load(checkPath)
    pre_epoch = stateDict['pre_epoch']
    model.load_state_dict(stateDict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(stateDict['optimizer_state_dict'])

    return pre_epoch, stateDict['performance'], \
           stateDict['model_state_dict'], stateDict['optimizer_state_dict']