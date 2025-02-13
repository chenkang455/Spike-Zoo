from torch.utils.data import Dataset
from pathlib import Path
from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import dataclass
import re


@dataclass
class REDS_SSIRConfig(BaseDatasetConfig):
    dataset_name: str = "reds_ssir"
    root_dir: Path = Path(__file__).parent.parent / Path("data/REDS_SSIR")
    train_width: int = 96
    train_height: int = 96
    test_width: int = 1280
    test_height: int = 720
    width: int = -1
    height: int = -1
    with_img: bool = True
    spike_length_train: int = 41
    spike_length_test: int = 301

    # post process
    def __post_init__(self):
        self.root_dir = Path(self.root_dir) if isinstance(self.root_dir, str) else self.root_dir
        # todo try download
        assert self.root_dir.exists(), f"No files found in {self.root_dir} for the specified dataset `{self.dataset_name}`."
        # train/test split
        if self.split == "train":
            self.spike_length = self.spike_length_train
            self.width = self.train_width
            self.height = self.train_height
        else:
            self.spike_length = self.spike_length_test
            self.width = self.test_width
            self.height = self.test_height


class REDS_SSIR(BaseDataset):
    def __init__(self, cfg: BaseDatasetConfig):
        super(REDS_SSIR, self).__init__(cfg)

    def prepare_data(self):
        """Specify the spike and image files to be loaded."""
        # spike/imgs dir train/test
        if self.cfg.split == "train":
            self.img_dir = self.cfg.root_dir / Path("crop_mini/spike/train/interp_20_alpha_0.40")
            self.spike_dir = self.cfg.root_dir / Path("crop_mini/image/train/train_orig")
        else:
            self.img_dir = self.cfg.root_dir / Path("imgs/val/val_orig")
            self.spike_dir = self.cfg.root_dir / Path("spike/val/interp_20_alpha_0.40")
        # get files
        self.spike_list = self.get_spike_files(self.spike_dir)
        self.img_list = []


class sreds_train(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pair_step = self.cfg["loader"]["pair_step"]
        self.augmentor = Augmentor(crop_size=self.cfg["loader"]["crop_size"])
        self.samples = self.collect_samples()
        print("The samples num of training data: {:d}".format(len(self.samples)))

    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1

    def collect_samples(self):
        spike_path = osp.join(
            self.cfg["data"]["root"], "crop_mini", "spike", "train", "interp_{:d}_alpha_{:.2f}".format(self.cfg["data"]["interp"], self.cfg["data"]["alpha"])
        )
        image_path = osp.join(self.cfg["data"]["root"], "crop_mini", "image", "train", "train_orig")
        scene_list = sorted(os.listdir(spike_path))
        samples = []

        for scene in scene_list:
            spike_dir = osp.join(spike_path, scene)
            image_dir = osp.join(image_path, scene)
            spk_path_list = sorted(os.listdir(spike_dir))

            spklen = len(spk_path_list)
            seq_len = self.cfg["model"]["seq_len"] + 2
            """
            for st in range(0, spklen - ((spklen - self.pair_step) % seq_len) - seq_len, self.pair_step):
                # 按照文件名称读取
                spikes_path_list = [osp.join(spike_dir, spk_path_list[ii]) for ii in range(st, st+seq_len)]
                images_path_list = [osp.join(image_dir, spk_path_list[ii][:-4]+'.png') for ii in range(st, st+seq_len)]
                
                if(self.confirm_exist([spikes_path_list, images_path_list])):
                    s = {}
                    s['spikes_paths'] = spikes_path_list
                    s['images_paths'] = images_path_list
                    samples.append(s)
            """
            # 按照文件名称读取
            spikes_path_list = [osp.join(spike_dir, spk_path_list[ii]) for ii in range(spklen)]
            images_path_list = [osp.join(image_dir, spk_path_list[ii][:-4] + ".png") for ii in range(spklen)]

            if self.confirm_exist([spikes_path_list, images_path_list]):
                s = {}
                s["spikes_paths"] = spikes_path_list
                s["images_paths"] = images_path_list
                samples.append(s)

        return samples

    def _load_sample(self, s):
        data = {}

        data["spikes"] = [np.array(dat_to_spmat(p, size=(96, 96)), dtype=np.float32) for p in s["spikes_paths"]]
        data["images"] = [read_img_gray(p) for p in s["images_paths"]]

        data["spikes"], data["images"] = self.augmentor(data["spikes"], data["images"])
        # print("data['spikes'][0].shape, data['images'][0].shape", data['spikes'][0].shape, data['images'][0].shape)

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


class sreds_test(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = self.collect_samples()
        print("The samples num of testing data: {:d}".format(len(self.samples)))

    def confirm_exist(self, path_list_list):
        for pl in path_list_list:
            for p in pl:
                if not osp.exists(p):
                    return 0
        return 1

    def collect_samples(self):
        spike_path = osp.join(
            self.cfg["data"]["root"], "spike", "val", "interp_{:d}_alpha_{:.2f}".format(self.cfg["data"]["interp"], self.cfg["data"]["alpha"])
        )
        image_path = osp.join(self.cfg["data"]["root"], "imgs", "val", "val_orig")
        scene_list = sorted(os.listdir(spike_path))
        samples = []

        for scene in scene_list:
            spike_dir = osp.join(spike_path, scene)
            image_dir = osp.join(image_path, scene)
            spk_path_list = sorted(os.listdir(spike_dir))

            spklen = len(spk_path_list)
            # seq_len = self.cfg['model']['seq_len']

            # 按照文件名称读取
            spikes_path_list = [osp.join(spike_dir, spk_path_list[ii]) for ii in range(spklen)]
            images_path_list = [osp.join(image_dir, spk_path_list[ii][:-4] + ".png") for ii in range(spklen)]

            if self.confirm_exist([spikes_path_list, images_path_list]):
                s = {}
                s["spikes_paths"] = spikes_path_list
                s["images_paths"] = images_path_list
                samples.append(s)

        return samples

    def _load_sample(self, s):
        data = {}
        data["spikes"] = [np.array(dat_to_spmat(p, size=(720, 1280)), dtype=np.float32) for p in s["spikes_paths"]]
        data["images"] = [read_img_gray(p) for p in s["images_paths"]]
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data
