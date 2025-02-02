import argparse
import glob
import os
import re
from collections import OrderedDict

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from nets import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Spk2ImgNet")
parser.add_argument(
    "--preprocess", type=bool, default=False, help="run prepare_data or not"
)
parser.add_argument("--batchSize", type=int, default=16, help="Trainning batch size")
parser.add_argument(
    "--num_of_layers", type=int, default=17, help="Number of total layers"
)
parser.add_argument("--epochs", type=int, default=61, help="Number of trainning epochs")
parser.add_argument(
    "--milestone",
    type=int,
    default=20,
    help="When to decay learning rate: should be less than epochs",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Initial learning rate; should be less than epochs",
)
parser.add_argument(
    "--outf",
    type=str,
    default="./ckpt2",
    help="path of log files",
)
parser.add_argument(
    "--load_model", type=bool, default=False, help="load model from net.pth"
)
opt = parser.parse_args()

if not os.path.exists(opt.outf):
    os.mkdir(opt.outf)


def find_last_checkpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, "model_*.pth"))
    if file_list:
        epoch_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epoch_exist.append(int(result[0]))
        initial_epoch = max(epoch_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def main():
    # Load dataset
    print("Loading dataset ...\n")
    dataset_train = Dataset("train")
    loader_train = DataLoader(
        dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True
    )
    print("# of training samples: %d\n" % int(len(dataset_train)))
    '''
    dataset_val = Dataset("val_stack")
    loader_val = DataLoader(
        dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=False
    ) 
    '''
    # Build model
    model = SpikeNet(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    if not opt.load_model:
        initial_epoch = 0
        print("haha")
    else:
        # load model
        initial_epoch = find_last_checkpoint(save_dir=opt.outf)
        print("load model from model.pth")
        state_dict = torch.load(
            os.path.join(opt.outf, "model_%03d.pth" % initial_epoch)
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    criterion = nn.L1Loss(size_average=True)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
    # Optimazer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    model.train()
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        avg_psnr = 0
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.0
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("learning rate %f" % current_lr)
        # train
        for i, (inputs, gt) in enumerate(loader_train, 0):
            # print(inputs.shape)
            inputs = Variable(inputs).cuda()
            gt = Variable(gt).cuda()
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            rec, est0, est1, est2, est3, est4 = model(inputs)
            est0 = est0 / 0.6
            est1 = est1 / 0.6
            est2 = est2 / 0.6
            est3 = est3 / 0.6
            est4 = est4 / 0.6
            rec = rec / 0.6
            loss = criterion(gt[:, 2:3, :, :], rec)
            for slice_id in range(4):
                loss = loss + 0.02 * (
                    criterion(gt[:, 0:1, :, :], est0[:, slice_id : slice_id + 1, :, :])
                    + criterion(
                        gt[:, 1:2, :, :], est1[:, slice_id : slice_id + 1, :, :]
                    )
                    + criterion(
                        gt[:, 2:3, :, :], est2[:, slice_id : slice_id + 1, :, :]
                    )
                    + criterion(
                        gt[:, 3:4, :, :], est3[:, slice_id : slice_id + 1, :, :]
                    )
                    + criterion(
                        gt[:, 4:5, :, :], est4[:, slice_id : slice_id + 1, :, :]
                    )
                )
            loss.backward()
            optimizer.step()
            rec = torch.clamp(rec, 0, 1)
            # print(rec)
            psnr_train = batch_psnr(rec, gt[:, 2:3, :, :], 1.0)
            # print(gt[:,2:3,:,:])
            avg_psnr += psnr_train
            if i % 10 == 0:
                print(
                    "[epoch %d][%d | %d] loss: %.4f PSNR_train: %.4f"
                    % (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train)
                )
            step += 1
        avg_psnr = avg_psnr / len(loader_train)
        print("avg_psnr: %.2f" % avg_psnr)

        if epoch % 5 == 0:
            '''
            # validate
            model.eval()
            psnr_val = 0
            for i, (inputs, gt) in enumerate(loader_val, 0):
                inputs = Variable(inputs).cuda()
                gt = Variable(gt).cuda()
                rec = model(inputs)
                rec = rec / 0.6
                rec = torch.clamp(rec, 0, 1)
                psnr_val += batch_psnr(rec, gt, 1.0)
            print("[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val / len(loader_val)))
            '''
            # save model
            torch.save(
                model.state_dict(),
                os.path.join(opt.outf, "model_%03d.pth" % (epoch + 1)),
            )


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(data_path="./Spk2ImgNet_train/train2/", patch_size=40, stride=40, h5_name='train')
    main()
