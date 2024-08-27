import cv2
import torch
import numpy as np
import imageio
import os
import torch.nn as nn
import random
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel 
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

def highlight_and_zoom_image(image_array, x, y, width, height):
    """
    返回一个包含原图（带有高亮区域标记）和保持比例的放大区域的合成图像。
    
    参数:
    image_array (numpy.ndarray): 原始图像数组。
    x (int): 放大区域左上角的 x 坐标。
    y (int): 放大区域左上角的 y 坐标。
    width (int): 放大区域的宽度。
    height (int): 放大区域的高度。
    
    返回:
    numpy.ndarray: 合成的图像。
    """
    x_end = min(x + width, image_array.shape[1])
    y_end = min(y + height, image_array.shape[0])
    width = x_end - x
    height = y_end - y
    
    zoomed_area = image_array[y:y_end, x:x_end]

    pil_image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_image)
    
    draw.rectangle([x, y, x_end, y_end], outline="green", width=2)  # 将边框颜色改为绿色

    pil_zoomed = Image.fromarray(zoomed_area)

    # 计算放大后的尺寸以保持比例
    scale_factor = pil_image.width / width
    new_height = int(height * scale_factor)
    
    pil_zoomed = pil_zoomed.resize((pil_image.width, new_height))

    total_height = pil_image.height + pil_zoomed.height
    new_image = Image.new('RGB', (pil_image.width, total_height))
    
    new_image.paste(pil_image, (0, 0))
    new_image.paste(pil_zoomed, (0, pil_image.height))  # 放在原图下方

    return np.array(new_image)



def load_network(load_path, network, strict=False):
    
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path)
    if load_path.endswith('.pt') == True:
        if isinstance(load_net, nn.DataParallel) or isinstance(load_net, DistributedDataParallel):
            load_net = load_net.module
        if isinstance(load_net, nn.Module):
            load_net = load_net.state_dict() 
    load_net_clean = OrderedDict()  
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    if 'model_state_dict' in load_net_clean.keys():
        network.load_state_dict(load_net_clean['model_state_dict'], strict=strict)
    elif 'model' in load_net_clean.keys():
        network.load_state_dict(load_net_clean['model'], strict=strict)
    else:
        network.load_state_dict(load_net_clean, strict=strict)
        
# Save Network 
def save_network(network, save_path):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def middleTFI(spike, middle, window):
    B, C, H, W = spike.shape
    lindex, rindex = torch.zeros([B,1, H, W]), torch.zeros([B,1,H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1):
        l = l - 1
        if l>=0:
            newpos = spike[:,l:l+1, :, :]*(1 - torch.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[:,r:r+1, :, :]*(1 - torch.sign(rindex))
            distance = r*newpos
            rindex += distance
    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval
    return tfi

def save_opt(opt,opt_path):
    with open(opt_path, 'w') as f:
        for key, value in vars(opt).items():
            f.write(f"{key}: {value}\n")

def save_gif(image_list, gif_path = 'test', duration = 2,RGB = True,nor = False):
    imgs = []
    with imageio.get_writer(os.path.join(gif_path + '.gif'), mode='I',duration = 1000 * duration / len(image_list),loop=0) as writer:
        for i in range(len(image_list)):
            img = normal_img(image_list[i],RGB,nor)
            writer.append_data(img)

def save_video(image_list,path = 'test',duration = 2,RGB = True,nor = False):
    os.makedirs('Video',exist_ok = True)
    imgs = []
    for i in range(len(image_list)):
        img = normal_img(image_list[i],RGB,nor)
        imgs.append(img)
    imageio.mimwrite(os.path.join('Video',path + '.mp4'), imgs, fps=30, quality=8)


def normal_img(img,RGB = True,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if (img.shape[0] == 3 or img.shape[0] == 1) and isinstance(img,torch.Tensor):
        img = img.permute(1,2,0)
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    if len(img.shape) == 2:
        img = img[...,None]
    if img.shape[-1] == 1:
        img = np.repeat(img,3,axis = -1)
    img = img.astype(np.uint8)
    if RGB == False:
        img = img[...,::-1]
    return img

def save_img(path = 'test.png',img = None,nor = True):
    if nor:
        img = 255 * ((img - img.min()) / (img.max() - img.min()))
    if isinstance(img,torch.Tensor):
        img = np.array(img.detach().cpu())
    img = img.astype(np.uint8)
    cv2.imwrite(path,img)

def make_folder(path):
    os.makedirs(path,exist_ok = True)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


def load_vidar_dat(filename, height = 250, width = 400):
    left_up=(0, 0)
    frame_cnt = None
    if isinstance(filename, str):
        array = np.fromfile(filename, dtype=np.uint8)
    elif isinstance(filename, (list, tuple)):
        l = []
        for name in filename:
            a = np.fromfile(name, dtype=np.uint8)
            l.append(a)
        array = np.concatenate(l)
    else:
        raise NotImplementedError

    window = (height - left_up[0], width - left_up[0])

    len_per_frame = height * width // 8
    framecnt = frame_cnt if frame_cnt != None else len(array) // len_per_frame

    spikes = []

    for i in range(framecnt):
        compr_frame = array[i * len_per_frame: (i + 1) * len_per_frame]
        blist = []
        for b in range(8):
            blist.append(np.right_shift(np.bitwise_and(compr_frame, np.left_shift(1, b)), b))
        
        frame_ = np.stack(blist).transpose()
        frame_ = np.flipud(frame_.reshape((height, width), order='C'))

        if window is not None:
            spk = frame_[left_up[0]:left_up[0] + window[0], left_up[1]:left_up[1] + window[1]]
        else:
            spk = frame_

        spk = spk.copy().astype(np.float32)[None]

        spikes.append(spk)

    return np.concatenate(spikes)


import logging
# log info
def setup_logging(log_file):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w')  # 使用'w'模式打开文件
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
    
    
def generate_labels(file_name):
    num_part = file_name.split('/')[-1]
    non_num_part = file_name.replace(num_part, '')
    num = int(num_part)
    labels = [non_num_part + str(num + 2 * i).zfill(len(num_part)) + '.png' for i in range(-3, 4)]
    return labels


def SpikeToRaw(save_path, SpikeSeq, filpud=True, delete_if_exists=True):
    """
        save spike sequence to .dat file
        save_path: full saving path (string)
        SpikeSeq: Numpy array (T x H x W)
        Rui Zhao
    """
    if delete_if_exists:
        if os.path.exists(save_path):
            os.remove(save_path)

    sfn, h, w = SpikeSeq.shape
    remainder = int((h * w) % 8)
    # assert (h * w) % 8 == 0
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        if filpud:
            # 模拟相机的倒像
            spike = np.flipud(SpikeSeq[img_id, :, :])
        else:
            spike = SpikeSeq[img_id, :, :]
        # numpy按自动按行排，数据也是按行存的
        # spike = spike.flatten()
        if remainder == 0:
            spike = spike.flatten()
        else:
            spike = np.concatenate([spike.flatten(), np.array([0]*(8-remainder))])
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())
    fid.close()
    return

