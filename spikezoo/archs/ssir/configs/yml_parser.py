import numpy as np
import torch
import yaml


class YAMLParser:
    """ 
        Modified from code from tudelft ssl-evflow
    """

    def __init__(self, config):
        self.reset_config()
        self.parse_config(config)
        # self.init_seeds()

    def parse_config(self, file):
        with open(file) as fid:
            yaml_config = yaml.load(fid, Loader=yaml.FullLoader)
        self.parse_dict(yaml_config)

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._device

    @property
    def loader_kwargs(self):
        return self._loader_kwargs

    def reset_config(self):
        self._config = {}

    def update(self, config):
        self.reset_config()
        self.parse_config(config)

    def parse_dict(self, input_dict, parent=None):
        if parent is None:
            parent = self._config
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in parent.keys():
                    parent[key] = {}
                self.parse_dict(val, parent[key])
            else:
                parent[key] = val

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # def init_seeds(self):
    #     torch.manual_seed(self._config["loader"]["seed"])
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(self._config["loader"]["seed"])
    #         torch.cuda.manual_seed_all(self._config["loader"]["seed"])

    def merge_configs(self, run):
        """
        Overwrites mlflow metadata with configs.
        """

        # parse mlflow settings
        config = {}
        for key in run.keys():
            if len(run[key]) > 0 and run[key][0] == "{":  # assume dictionary
                config[key] = eval(run[key])
            else:  # string
                config[key] = run[key]

        # overwrite with config settings
        self.parse_dict(self._config, config)
        self.combine_entries(config)

        return config