import logging
from dataclasses import dataclass, field, asdict
import requests
from tqdm import tqdm
import os
import torch
import numpy as np
import random

# log info
def setup_logging(log_file):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w")  # 使用'w'模式打开文件
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def save_config(cfg, filename, mode="w"):
    """Save the config file to the given filename."""
    filename = str(filename)
    with open(filename, mode) as file:
        for key, value in asdict(cfg).items():
            file.write(f"{key} = {value}\n")
        file.write("\n")


def download_file(url, output_path):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Accept": "*/*",  
        "Accept-Encoding": "gzip, deflate, br", 
        "Connection": "keep-alive"  
    }

    try:
        print(f"Ready to download the file from {url} 😊😊😊.")
        response = requests.head(url, headers=headers)
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))

        response = requests.get(url, stream=True)
        response.raise_for_status()

        if response.status_code == 200:
            with open(output_path, "wb") as file, tqdm(desc="Downloading", total=file_size, unit="B", unit_scale=True) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))

            print(f"Files downloaded successfully 🎉🎉🎉 and saved on {output_path}!")
        else:
            raise RuntimeError(f"Files fail to download 😔😔😔. Try downloading it from {url} and move it to {output_path}.")
    
    except requests.exceptions.RequestException as e:
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Partial download failed. The incomplete file has been removed. 😔😔😔")
        raise RuntimeError(f"Files fail to download 😔😔😔. Try downloading it from {url} and move it to {output_path}.")
    
def check_file_exists(url):
    response = requests.head(url)
    if response.status_code == 200:
        return True
    else:
        return False

def getattr_case_insensitive(obj, name):
    name = name.lower()
    for attr in dir(obj):
        if attr.lower() == name:
            return getattr(obj, attr)
    raise RuntimeError("No attr found!!")


def set_random_seed(seed):
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)