import importlib
import inspect
from spikezoo.models.base_model import BaseModel,BaseModelConfig
import os
from pathlib import Path

current_file_path = Path(__file__).parent
files_list = os.listdir(os.path.dirname(os.path.abspath(__file__)))
model_list = [file.split("_")[0] for file in files_list if file.endswith("_model.py")]

# todo register function
def build_model_cfg(cfg: BaseModelConfig):
    """Build the model from the given model config."""
    # model module name
    module_name = cfg.model_name + "_model"
    assert cfg.model_name in model_list, f"Given model {cfg.model_name} not in our model zoo {model_list}."
    module_name = "spikezoo.models." + module_name
    module = importlib.import_module(module_name)
    # model,model_config
    classes = sorted([name for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == module.__name__])
    model_cls: BaseModel = getattr(module, classes[0])
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
    classes = sorted([name for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == module.__name__])
    model_cls: BaseModel = getattr(module, classes[0])
    model_cfg: BaseModelConfig = getattr(module, classes[1])()
    model = model_cls(model_cfg)
    return model