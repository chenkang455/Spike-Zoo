from utils import *
import glob
from torchvision import transforms
from tqdm import trange
import random
from torchvision import transforms
from torchvision.transforms import functional as TF

class SpikeData_REDS(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_type = 'REDS', stage = 'train',
                 spike_dir_name = 'spike',return_seq = False,augument = True):
        self.root_dir = root_dir
        self.data_type = data_type
        self.spike_dir_name = spike_dir_name
        self.return_seq = return_seq
        self.augument = augument
        self.stage = stage
        if data_type == 'REDS':
            pattern = os.path.join(self.root_dir, stage,spike_dir_name,'*')
            self.spike_list = sorted(glob.glob(pattern))
            self.width = 400
            self.height = 250
        self.length = len(self.spike_list)
        self.transform = RandomTransforms(p_horizontal=0.5,p_vertical=0.5,degrees=30)
    def __len__(self):
        return self.length
        
    def __getitem__(self, index: int):
        # blur and spike load
        spike_name = self.spike_list[index]
        spike = load_vidar_dat(spike_name,width=self.width ,height=self.height )
        spike = torch.from_numpy(spike)

        # sharp load
        if self.return_seq == False or self.stage == 'test':
            sharp_name = spike_name.replace('.dat','.png').replace(self.spike_dir_name,'gt')
            sharp = cv2.imread(sharp_name)
            sharp = 0.11 * sharp[...,0:1] + 0.59 * sharp[...,1:2] + 0.3 * sharp[...,2:3]
            sharp = torch.from_numpy(sharp).permute((2,0,1)).float() / 255
        else:
            sharp_name = spike_name.replace('.dat','.png').replace(self.spike_dir_name,'gt')
            sharp_name_list = [sharp_name.replace('id21','id' + str(i)) for i in [7,14,21,28,35]]
            sharp_list = []
            for sharp_name in sharp_name_list:
                sharp = cv2.imread(sharp_name)
                sharp = 0.11 * sharp[...,0:1] + 0.59 * sharp[...,1:2] + 0.3 * sharp[...,2:3]
                sharp = torch.from_numpy(sharp).permute((2,0,1)).float() / 255
                sharp_list.append(sharp)
            sharp = torch.cat(sharp_list)
        
        if self.augument and self.stage == 'train':
            spike,sharp = self.transform(spike,sharp)
        
        return spike,sharp


class SpikeData_Real(torch.utils.data.Dataset):
    def __init__(self,root_dir):
        self.spike_list = [] 
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.dat'):
                    self.spike_list.append(os.path.join(root, file))
        self.length = len(self.spike_list)
        self.width = 400
        self.height = 250
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index: int):
        spike_name = self.spike_list[index]
        spike = load_vidar_dat(spike_name,width=self.width ,height=self.height)
        spike = torch.from_numpy(spike)
        _  = torch.zeros((1,1,self.height,self.width))
        return spike,_
    
class SpikeData_UHSR(torch.utils.data.Dataset):
    def __init__(self, root_dir, stage):
        self.root_dir = root_dir
        self.stage = stage
        self.data_list = os.path.join(root_dir, stage)
        self.data_list = sorted(os.listdir(self.data_list))
        self.length = len(self.data_list)
        
    
    def __getitem__(self, idx: int):
        data = np.load(os.path.join(self.root_dir,self.stage,self.data_list[idx]))
        spk = data['spk'].astype(np.float32)
        spk = spk[:,13:237,13:237] # [200,250,250] -> [200,224,224]
        return spk,np.zeros(1)

    def __len__(self):
        return self.length


class RandomTransforms:
    def __init__(self,p_horizontal = 0.5,p_vertical=0.5,degrees = 30):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
        self.degrees = degrees

    def __call__(self, img, target):
        if random.random() < self.p_horizontal:
            img = TF.hflip(img)
            target = TF.hflip(target)

        if random.random() < self.p_vertical:
            img = TF.vflip(img)
            target = TF.vflip(target)

        angle = random.uniform(-self.degrees, self.degrees)
        img = TF.rotate(img, angle)
        target = TF.rotate(target, angle)

        return img, target

if __name__ == "__main__":
    pass