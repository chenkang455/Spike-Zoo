import unittest
import tempfile
import os
from pathlib import Path
from spikezoo.config.pipeline_configs import (
    PipelineConfig, 
    TrainPipelineConfig, 
    EnsemblePipelineConfig,
    load_pipeline_config
)


class TestPipelineConfigs(unittest.TestCase):
    """PipelineConfigs unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Test cleanup."""
        # Remove temporary directory
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.temp_dir)
    
    def test_pipeline_config_creation(self):
        """Test PipelineConfig creation."""
        config = PipelineConfig(
            version="local",
            save_folder="results",
            exp_name="test_exp",
            save_metric=True,
            metric_names=["psnr", "ssim"],
            save_img=True,
            img_norm=False,
            bs_test=1,
            nw_test=0,
            pin_memory=False,
            _mode="single_mode"
        )
        
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "results")
        self.assertEqual(config.exp_name, "test_exp")
        self.assertTrue(config.save_metric)
        self.assertEqual(config.metric_names, ["psnr", "ssim"])
        self.assertTrue(config.save_img)
        self.assertFalse(config.img_norm)
        self.assertEqual(config.bs_test, 1)
        self.assertEqual(config.nw_test, 0)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config._mode, "single_mode")
    
    def test_pipeline_config_from_dict(self):
        """Test PipelineConfig creation from dictionary."""
        config_dict = {
            "version": "v010",
            "save_folder": "outputs",
            "exp_name": "dict_test",
            "save_metric": False,
            "metric_names": ["psnr"],
            "save_img": False,
            "img_norm": True,
            "bs_test": 2,
            "nw_test": 1,
            "pin_memory": True,
            "_mode": "multi_mode"
        }
        
        config = PipelineConfig.from_dict(config_dict)
        
        self.assertEqual(config.version, "v010")
        self.assertEqual(config.save_folder, "outputs")
        self.assertEqual(config.exp_name, "dict_test")
        self.assertFalse(config.save_metric)
        self.assertEqual(config.metric_names, ["psnr"])
        self.assertFalse(config.save_img)
        self.assertTrue(config.img_norm)
        self.assertEqual(config.bs_test, 2)
        self.assertEqual(config.nw_test, 1)
        self.assertTrue(config.pin_memory)
        self.assertEqual(config._mode, "multi_mode")
    
    def test_train_pipeline_config_creation(self):
        """Test TrainPipelineConfig creation."""
        config = TrainPipelineConfig(
            version="local",
            save_folder="results",
            exp_name="train_test",
            epochs=50,
            steps_per_save_imgs=5,
            steps_per_save_ckpt=5,
            steps_per_cal_metrics=5,
            steps_grad_accumulation=2,
            _mode="train_mode",
            use_tensorboard=True,
            seed=42,
            bs_train=4,
            nw_train=2,
            loss_weight_dict={"l1": 0.5}
        )
        
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "results")
        self.assertEqual(config.exp_name, "train_test")
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.steps_per_save_imgs, 5)
        self.assertEqual(config.steps_per_save_ckpt, 5)
        self.assertEqual(config.steps_per_cal_metrics, 5)
        self.assertEqual(config.steps_grad_accumulation, 2)
        self.assertEqual(config._mode, "train_mode")
        self.assertTrue(config.use_tensorboard)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.bs_train, 4)
        self.assertEqual(config.nw_train, 2)
        self.assertEqual(config.loss_weight_dict, {"l1": 0.5})
    
    def test_train_pipeline_config_from_dict(self):
        """Test TrainPipelineConfig creation from dictionary."""
        config_dict = {
            "version": "v023",
            "save_folder": "training_results",
            "exp_name": "dict_train_test",
            "epochs": 100,
            "steps_per_save_imgs": 10,
            "steps_per_save_ckpt": 10,
            "steps_per_cal_metrics": 10,
            "steps_grad_accumulation": 4,
            "_mode": "train_mode",
            "use_tensorboard": False,
            "seed": 123,
            "bs_train": 8,
            "nw_train": 4,
            "optimizer_cfg": {
                "type": "adam",
                "lr": 0.001
            },
            "loss_weight_dict": {
                "l1": 1.0,
                "l2": 0.1
            }
        }
        
        config = TrainPipelineConfig.from_dict(config_dict)
        
        self.assertEqual(config.version, "v023")
        self.assertEqual(config.save_folder, "training_results")
        self.assertEqual(config.exp_name, "dict_train_test")
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.steps_per_save_imgs, 10)
        self.assertEqual(config.steps_per_save_ckpt, 10)
        self.assertEqual(config.steps_per_cal_metrics, 10)
        self.assertEqual(config.steps_grad_accumulation, 4)
        self.assertEqual(config._mode, "train_mode")
        self.assertFalse(config.use_tensorboard)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.bs_train, 8)
        self.assertEqual(config.nw_train, 4)
        self.assertEqual(config.loss_weight_dict, {"l1": 1.0, "l2": 0.1})
    
    def test_ensemble_pipeline_config_creation(self):
        """Test EnsemblePipelineConfig creation."""
        config = EnsemblePipelineConfig(
            version="local",
            save_folder="results",
            exp_name="ensemble_test",
            save_metric=True,
            metric_names=["psnr", "ssim", "lpips"],
            save_img=True,
            img_norm=False,
            bs_test=1,
            nw_test=0,
            pin_memory=False,
            _mode="multi_mode"
        )
        
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "results")
        self.assertEqual(config.exp_name, "ensemble_test")
        self.assertTrue(config.save_metric)
        self.assertEqual(config.metric_names, ["psnr", "ssim", "lpips"])
        self.assertTrue(config.save_img)
        self.assertFalse(config.img_norm)
        self.assertEqual(config.bs_test, 1)
        self.assertEqual(config.nw_test, 0)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config._mode, "multi_mode")
    
    def test_ensemble_pipeline_config_from_dict(self):
        """Test EnsemblePipelineConfig creation from dictionary."""
        config_dict = {
            "version": "local",
            "save_folder": "ensemble_results",
            "exp_name": "dict_ensemble_test",
            "_mode": "multi_mode",
            "metric_names": ["psnr", "ssim", "niqe", "brisque"],
            "save_img": True
        }
        
        config = EnsemblePipelineConfig.from_dict(config_dict)
        
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "ensemble_results")
        self.assertEqual(config.exp_name, "dict_ensemble_test")
        self.assertEqual(config._mode, "multi_mode")
        self.assertEqual(config.metric_names, ["psnr", "ssim", "niqe", "brisque"])
        self.assertTrue(config.save_img)
    
    def test_load_pipeline_config_base(self):
        """Test loading base pipeline configuration from file."""
        # Create a test config file
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "loaded_test",
            "save_metric": True,
            "metric_names": ["psnr", "ssim"],
            "_mode": "single_mode"
        }
        
        config_path = Path(self.temp_dir) / "base_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = load_pipeline_config(str(config_path), "base")
        
        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "results")
        self.assertEqual(config.exp_name, "loaded_test")
    
    def test_load_pipeline_config_train(self):
        """Test loading train pipeline configuration from file."""
        # Create a test config file
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "loaded_train_test",
            "epochs": 50,
            "_mode": "train_mode",
            "use_tensorboard": True
        }
        
        config_path = Path(self.temp_dir) / "train_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = load_pipeline_config(str(config_path), "train")
        
        self.assertIsInstance(config, TrainPipelineConfig)
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "results")
        self.assertEqual(config.exp_name, "loaded_train_test")
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config._mode, "train_mode")
        self.assertTrue(config.use_tensorboard)
    
    def test_load_pipeline_config_ensemble(self):
        """Test loading ensemble pipeline configuration from file."""
        # Create a test config file
        config_data = {
            "version": "local",
            "save_folder": "results",
            "exp_name": "loaded_ensemble_test",
            "_mode": "multi_mode",
            "metric_names": ["psnr", "ssim", "niqe"]
        }
        
        config_path = Path(self.temp_dir) / "ensemble_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = load_pipeline_config(str(config_path), "ensemble")
        
        self.assertIsInstance(config, EnsemblePipelineConfig)
        self.assertEqual(config.version, "local")
        self.assertEqual(config.save_folder, "results")
        self.assertEqual(config.exp_name, "loaded_ensemble_test")
        self.assertEqual(config._mode, "multi_mode")
        self.assertEqual(config.metric_names, ["psnr", "ssim", "niqe"])


if __name__ == '__main__':
    unittest.main()