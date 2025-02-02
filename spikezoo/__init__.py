from .utils.spike_utils import load_vidar_dat
from .models import model_list
from .datasets import dataset_list
from .metrics import metric_all_names

def get_datasets():
    return dataset_list

def get_models():
    return model_list

def get_metrics():
    return metric_all_names