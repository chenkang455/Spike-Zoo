from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig
from dataclasses import replace
import importlib, inspect
import os
import torch
from typing import Literal

# todo auto detect/register datasets
files_list = os.listdir(os.path.dirname(os.path.abspath(__file__)))
dataset_list = [file.replace("_dataset.py", "") for file in files_list if file.endswith("_dataset.py")]

# todo register function
def build_dataset_cfg(cfg: BaseDatasetConfig, split: Literal["train", "test"] = "test"):
    """Build the dataset from the given dataset config."""
    # build new cfg according to split
    cfg = replace(cfg,split = split,spike_length = cfg.spike_length_train if split == "train" else cfg.spike_length_test)
    # dataset module
    module_name = cfg.dataset_name + "_dataset"
    assert cfg.dataset_name in dataset_list, f"Given dataset {cfg.dataset_name} not in our dataset list {dataset_list}."
    module_name = "spikezoo.datasets." + module_name
    module = importlib.import_module(module_name)
    # dataset,dataset_config
    classes = sorted([name for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == module.__name__])
    dataset_cls: BaseDataset = getattr(module, classes[0])
    dataset = dataset_cls(cfg)
    return dataset


def build_dataset_name(dataset_name: str, split: Literal["train", "test"] = "test"):
    """Build the default dataset from the given name."""
    module_name = dataset_name + "_dataset"
    assert dataset_name in dataset_list, f"Given dataset {dataset_name} not in our dataset list {dataset_list}."
    module_name = "spikezoo.datasets." + module_name
    module = importlib.import_module(module_name)
    # dataset,dataset_config
    classes = sorted([name for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == module.__name__])
    dataset_cls: BaseDataset = getattr(module, classes[0])
    dataset_cfg: BaseDatasetConfig = getattr(module, classes[1])(split=split)
    dataset = dataset_cls(dataset_cfg)
    return dataset


# todo to modify according to the basicsr
def build_dataloader(dataset: BaseDataset,cfg = None):
    # train dataloader
    if dataset.cfg.split == "train":
        if cfg is None:
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=cfg.bs_train, shuffle=True, num_workers=cfg.num_workers,pin_memory=cfg.pin_memory)
    # test dataloader
    elif dataset.cfg.split == "test":
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


# dataset_size_dict = {}
# for dataset in dataset_list:
#     module_name = dataset + "_dataset"
#     module_name = "spikezoo.datasets." + module_name
#     module = importlib.import_module(module_name)
#     classes = sorted([name for name, obj in inspect.getmembers(module) if inspect.isclass(obj)])
#     dataset_cfg: BaseDatasetConfig = getattr(module, classes[1])
#     dataset_size_dict[dataset] = (dataset_cfg.height, dataset_cfg.width)


# def get_dataset_size(name):
#     assert name in dataset_list, f"Given dataset {name} not in our dataset list {dataset_list}."
#     return dataset_size_dict[name]
