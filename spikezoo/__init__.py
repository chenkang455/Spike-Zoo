from .utils.spike_utils import *
from .datasets import dataset_list
from .metrics import metric_all_names

# Import model list differently to avoid circular imports
import spikezoo.models
model_list = spikezoo.models.__all__

# METHOD NAME DEFINITION
METHODS = model_list
class METHOD:
    BASE = "base"
    TFP = "tfp"
    TFI = "tfi"
    SPK2IMGNET = "spk2imgnet"
    WGSE = "wgse"
    SSML = "ssml"
    BSF = "bsf"
    STIR = "stir"
    SSIR = "ssir"
    SPIKECLIP = "spikeclip"

# DATASET NAME DEFINITION
DATASETS = dataset_list
class DATASET:
    BASE = "base"
    REDS_BASE = "reds_base"
    REALWORLD = "realworld"
    UHSR = "uhsr"

# METRIC NAME DEFINITION
METRICS = metric_all_names