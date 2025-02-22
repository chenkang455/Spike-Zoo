import importlib
import inspect
from spikezoo.models.base_model import BaseModel,BaseModelConfig

from spikezoo.utils.other_utils import getattr_case_insensitive
import os
from pathlib import Path

current_file_path = Path(__file__).parent
files_list = os.listdir(os.path.dirname(os.path.abspath(__file__)))
model_list = [file.split("_")[0] for file in files_list if file.endswith("_model.py")]

# todo register function
def build_model_cfg(cfg: BaseModelConfig):
    """Build the model from the given model config."""
    # model module name
    if cfg.model_cls_local == None:
        module_name = cfg.model_name + "_model"
        assert cfg.model_name in model_list, f"Given model {cfg.model_name} not in our model zoo {model_list}."
        module_name = "spikezoo.models." + module_name
        module = importlib.import_module(module_name)
        # model,model_config
        model_name = cfg.model_name
        model_name = model_name + 'Model' if model_name == "base" else model_name
        model_cls: BaseModel = getattr_case_insensitive(module,model_name)
    else:
        model_cls: BaseModel = cfg.model_cls_local
    model = model_cls(cfg)
    return model

def build_model_name(model_name: str):
    """Build the default dataset from the given name."""
    # model module name
    module_name = model_name + "_model"
    assert model_name in model_list, f"Given model {model_name} not in our model zoo {model_list}."
    module_name = "spikezoo.models." + module_name
    module = importlib.import_module(module_name)
    # model,model_config
    model_name = model_name + 'Model' if model_name == "base" else model_name
    model_cls: BaseModel = getattr_case_insensitive(module,model_name)
    model_cfg: BaseModelConfig = getattr_case_insensitive(module, model_name + 'config')()
    model = model_cls(model_cfg)
    return model
