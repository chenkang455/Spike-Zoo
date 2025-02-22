from spikezoo.datasets.base_dataset import BaseDataset, BaseDatasetConfig

from dataclasses import replace
import importlib, inspect
import os
import torch
from typing import Literal
from spikezoo.utils.other_utils import getattr_case_insensitive

# todo auto detect/register datasets
files_list = os.listdir(os.path.dirname(os.path.abspath(__file__)))
dataset_list = [file.replace("_dataset.py", "") for file in files_list if file.endswith("_dataset.py")]


# todo register function
def build_dataset_cfg(cfg: BaseDatasetConfig):
    """Build the dataset from the given dataset config."""
    # dataset module
    if cfg.dataset_cls_local == None:
        module_name = cfg.dataset_name + "_dataset"
        assert cfg.dataset_name in dataset_list, f"Given dataset {cfg.dataset_name} not in our dataset list {dataset_list}."
        module_name = "spikezoo.datasets." + module_name
        module = importlib.import_module(module_name)
        # dataset,dataset_config
        dataset_name = cfg.dataset_name
        dataset_name = dataset_name + "Dataset" if dataset_name == "base" else dataset_name
        dataset_cls: BaseDataset = getattr_case_insensitive(module, dataset_name)
    else:
        dataset_cls = cfg.dataset_cls_local
    dataset = dataset_cls(cfg)
    return dataset

def build_dataset_name(dataset_name: str):
    """Build the default dataset from the given name."""
    module_name = dataset_name + "_dataset"
    assert dataset_name in dataset_list, f"Given dataset {dataset_name} not in our dataset list {dataset_list}."
    module_name = "spikezoo.datasets." + module_name
    module = importlib.import_module(module_name)
    # dataset,dataset_config
    dataset_name = dataset_name + "Dataset" if dataset_name == "base" else dataset_name
    dataset_cls: BaseDataset = getattr_case_insensitive(module, dataset_name)
    dataset_cfg: BaseDatasetConfig = getattr_case_insensitive(module, dataset_name + "config")()
    dataset = dataset_cls(dataset_cfg)
    return dataset


# todo to modify according to the basicsr
def build_dataloader(dataset, cfg):
    # train dataloader
    if dataset.split == "train" and cfg._mode == "train_mode":
        return torch.utils.data.DataLoader(dataset, batch_size=cfg.bs_train, shuffle=True, num_workers=cfg.nw_train, pin_memory=cfg.pin_memory)
    # test dataloader
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=cfg.bs_test, shuffle=False, num_workers=cfg.nw_test,pin_memory=False)


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
