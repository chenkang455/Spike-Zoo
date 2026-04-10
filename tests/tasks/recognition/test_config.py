import unittest
from src.tasks.recognition.config import RecognitionModelConfig, RecognitionDatasetConfig


class TestRecognitionConfig(unittest.TestCase):
    def test_recognition_model_config_defaults(self):
        config = RecognitionModelConfig()
        self.assertEqual(config.model_type, "dmer_net18")
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.T, 7)
        self.assertFalse(config.pretrained)
        self.assertIsNone(config.checkpoint_path)
        self.assertEqual(config.model_kwargs, {})

    def test_recognition_model_config_custom_values(self):
        config = RecognitionModelConfig(
            model_type="dmer_net34",
            num_classes=100,
            T=5,
            pretrained=True,
            checkpoint_path="/path/to/checkpoint.pth",
            model_kwargs={"dropout": 0.5}
        )
        self.assertEqual(config.model_type, "dmer_net34")
        self.assertEqual(config.num_classes, 100)
        self.assertEqual(config.T, 5)
        self.assertTrue(config.pretrained)
        self.assertEqual(config.checkpoint_path, "/path/to/checkpoint.pth")
        self.assertEqual(config.model_kwargs, {"dropout": 0.5})

    def test_recognition_dataset_config_defaults(self):
        config = RecognitionDatasetConfig()
        self.assertEqual(config.dataset_type, "dmer")
        self.assertEqual(config.T, 7)
        self.assertEqual(config.train_split, 0.8)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.shuffle)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.transform_config, {})

    def test_recognition_dataset_config_custom_values(self):
        config = RecognitionDatasetConfig(
            dataset_type="rps",
            data_path="/path/to/data",
            T=5,
            train_split=0.9,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            transform_config={"normalize": True}
        )
        self.assertEqual(config.dataset_type, "rps")
        self.assertEqual(config.data_path, "/path/to/data")
        self.assertEqual(config.T, 5)
        self.assertEqual(config.train_split, 0.9)
        self.assertEqual(config.batch_size, 16)
        self.assertFalse(config.shuffle)
        self.assertEqual(config.num_workers, 2)
        self.assertEqual(config.transform_config, {"normalize": True})


if __name__ == '__main__':
    unittest.main()