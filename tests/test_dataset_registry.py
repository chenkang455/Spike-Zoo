import unittest
import sys
import os
import torch
from torch.utils.data import Dataset

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.core.dataset_registry import (
    DatasetRegistry,
    DatasetInfo,
    register_dataset,
    unregister_dataset,
    get_dataset_info,
    list_datasets,
    create_dataset,
    create_dataset_with_config,
    get_dataset_registry,
    discover_datasets_from_directory
)


class TestDatasetInfo(unittest.TestCase):
    """DatasetInfo unit tests."""
    
    def test_dataset_info_creation(self):
        """Test DatasetInfo creation."""
        dataset_info = DatasetInfo(
            name="test_dataset",
            version="1.0.0",
            description="Test dataset",
            author="Test Author",
            category="test",
            tags=["tag1", "tag2"]
        )
        
        self.assertEqual(dataset_info.name, "test_dataset")
        self.assertEqual(dataset_info.version, "1.0.0")
        self.assertEqual(dataset_info.description, "Test dataset")
        self.assertEqual(dataset_info.author, "Test Author")
        self.assertEqual(dataset_info.category, "test")
        self.assertEqual(dataset_info.tags, ["tag1", "tag2"])
        self.assertIsNone(dataset_info.config_class)
        self.assertIsNone(dataset_info.dataset_class)
        self.assertIsNone(dataset_info.factory_function)


class TestDatasetRegistry(unittest.TestCase):
    """DatasetRegistry unit tests."""
    
    def setUp(self):
        """Test setup."""
        self.registry = DatasetRegistry()
    
    def test_register_dataset_with_class(self):
        """Test registering dataset with class."""
        class TestDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(
            name="test_dataset",
            dataset_class=TestDataset,
            version="1.0.0",
            description="Test dataset with class"
        )
        
        # Check registration
        self.assertIn("test_dataset", self.registry.datasets)
        dataset_info = self.registry.get_dataset_info("test_dataset")
        self.assertIsNotNone(dataset_info)
        self.assertEqual(dataset_info.name, "test_dataset")
        self.assertEqual(dataset_info.dataset_class, TestDataset)
    
    def test_register_dataset_with_factory(self):
        """Test registering dataset with factory function."""
        def create_dataset():
            class SimpleDataset(Dataset):
                def __init__(self):
                    self.data = [1, 2, 3]
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            return SimpleDataset()
        
        self.registry.register_dataset(
            name="factory_dataset",
            factory_function=create_dataset,
            version="1.0.0",
            description="Test dataset with factory"
        )
        
        # Check registration
        self.assertIn("factory_dataset", self.registry.datasets)
        self.assertIn("factory_dataset", self.registry.factories)
        dataset_info = self.registry.get_dataset_info("factory_dataset")
        self.assertIsNotNone(dataset_info)
        self.assertEqual(dataset_info.name, "factory_dataset")
        self.assertEqual(dataset_info.factory_function, create_dataset)
    
    def test_unregister_dataset(self):
        """Test unregistering dataset."""
        class TestDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="test_dataset", dataset_class=TestDataset)
        self.assertIn("test_dataset", self.registry.datasets)
        
        self.registry.unregister_dataset("test_dataset")
        self.assertNotIn("test_dataset", self.registry.datasets)
        self.assertNotIn("test_dataset", self.registry.factories)
    
    def test_get_dataset_info(self):
        """Test getting dataset info."""
        class TestDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(
            name="test_dataset",
            dataset_class=TestDataset,
            description="Test dataset"
        )
        
        # Get existing dataset info
        dataset_info = self.registry.get_dataset_info("test_dataset")
        self.assertIsNotNone(dataset_info)
        self.assertEqual(dataset_info.name, "test_dataset")
        self.assertEqual(dataset_info.description, "Test dataset")
        
        # Get non-existent dataset info
        dataset_info = self.registry.get_dataset_info("non_existent")
        self.assertIsNone(dataset_info)
    
    def test_list_datasets(self):
        """Test listing datasets."""
        class TestDataset1(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestDataset2(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [4, 5, 6]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="dataset1", dataset_class=TestDataset1)
        self.registry.register_dataset(name="dataset2", dataset_class=TestDataset2)
        
        datasets = self.registry.list_datasets()
        self.assertIn("dataset1", datasets)
        self.assertIn("dataset2", datasets)
        self.assertEqual(len(datasets), 2)
    
    def test_list_datasets_by_category(self):
        """Test listing datasets by category."""
        class TestDataset1(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestDataset2(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [4, 5, 6]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="dataset1", dataset_class=TestDataset1, category="classification")
        self.registry.register_dataset(name="dataset2", dataset_class=TestDataset2, category="regression")
        
        classification_datasets = self.registry.list_datasets_by_category("classification")
        regression_datasets = self.registry.list_datasets_by_category("regression")
        
        self.assertIn("dataset1", classification_datasets)
        self.assertNotIn("dataset2", classification_datasets)
        self.assertIn("dataset2", regression_datasets)
        self.assertNotIn("dataset1", regression_datasets)
    
    def test_list_datasets_by_tag(self):
        """Test listing datasets by tag."""
        class TestDataset1(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestDataset2(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [4, 5, 6]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="dataset1", dataset_class=TestDataset1, tags=["image", "cnn"])
        self.registry.register_dataset(name="dataset2", dataset_class=TestDataset2, tags=["text", "rnn"])
        self.registry.register_dataset(name="dataset3", dataset_class=TestDataset1, tags=["image", "gan"])
        
        image_datasets = self.registry.list_datasets_by_tag("image")
        text_datasets = self.registry.list_datasets_by_tag("text")
        
        self.assertIn("dataset1", image_datasets)
        self.assertIn("dataset3", image_datasets)
        self.assertNotIn("dataset2", image_datasets)
        self.assertIn("dataset2", text_datasets)
        self.assertNotIn("dataset1", text_datasets)
        self.assertNotIn("dataset3", text_datasets)
    
    def test_create_dataset_with_class(self):
        """Test creating dataset with class."""
        class TestDataset(Dataset):
            def __init__(self, size=100):
                super().__init__()
                self.size = size
                self.data = list(range(size))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="test_dataset", dataset_class=TestDataset)
        
        # Create dataset
        dataset = self.registry.create_dataset("test_dataset", size=50)
        self.assertIsInstance(dataset, TestDataset)
        self.assertEqual(dataset.size, 50)
        self.assertEqual(len(dataset), 50)
    
    def test_create_dataset_with_factory(self):
        """Test creating dataset with factory function."""
        def create_dataset(size=100):
            class SimpleDataset(Dataset):
                def __init__(self, size):
                    self.size = size
                    self.data = list(range(size))
                
                def __len__(self):
                    return self.size
                
                def __getitem__(self, idx):
                    return self.data[idx]
            return SimpleDataset(size)
        
        self.registry.register_dataset(name="factory_dataset", factory_function=create_dataset)
        
        # Create dataset
        dataset = self.registry.create_dataset("factory_dataset", 75)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 75)
    
    def test_create_dataset_with_config(self):
        """Test creating dataset with config."""
        class TestDataset(Dataset):
            def __init__(self, config=None):
                super().__init__()
                if config:
                    self.size = config.size
                else:
                    self.size = 100
                self.data = list(range(self.size))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestConfig:
            def __init__(self, size=100):
                self.size = size
        
        self.registry.register_dataset(name="config_dataset", dataset_class=TestDataset)
        
        # Create dataset with config
        config = TestConfig(size=200)
        dataset = self.registry.create_dataset_with_config("config_dataset", config)
        self.assertIsInstance(dataset, TestDataset)
        self.assertEqual(dataset.size, 200)
        self.assertEqual(len(dataset), 200)
    
    def test_create_dataset_not_found(self):
        """Test creating non-existent dataset."""
        with self.assertRaises(ValueError):
            self.registry.create_dataset("non_existent_dataset")
    
    def test_create_dataset_no_creation_method(self):
        """Test creating dataset without creation method."""
        self.registry.register_dataset(name="incomplete_dataset", description="No class or factory")
        
        with self.assertRaises(ValueError):
            self.registry.create_dataset("incomplete_dataset")
    
    def test_get_dataset_categories(self):
        """Test getting dataset categories."""
        class TestDataset1(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestDataset2(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [4, 5, 6]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="dataset1", dataset_class=TestDataset1, category="classification")
        self.registry.register_dataset(name="dataset2", dataset_class=TestDataset2, category="regression")
        self.registry.register_dataset(name="dataset3", dataset_class=TestDataset1, category="classification")
        
        categories = self.registry.get_dataset_categories()
        self.assertIn("classification", categories)
        self.assertIn("regression", categories)
        self.assertEqual(len(categories), 2)
    
    def test_get_dataset_tags(self):
        """Test getting dataset tags."""
        class TestDataset1(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestDataset2(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [4, 5, 6]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        self.registry.register_dataset(name="dataset1", dataset_class=TestDataset1, tags=["cnn", "image"])
        self.registry.register_dataset(name="dataset2", dataset_class=TestDataset2, tags=["rnn", "text"])
        self.registry.register_dataset(name="dataset3", dataset_class=TestDataset1, tags=["cnn", "audio"])
        
        tags = self.registry.get_dataset_tags()
        self.assertIn("cnn", tags)
        self.assertIn("image", tags)
        self.assertIn("rnn", tags)
        self.assertIn("text", tags)
        self.assertIn("audio", tags)
        self.assertEqual(len(tags), 5)


class TestGlobalFunctions(unittest.TestCase):
    """Test global dataset registry functions."""
    
    def setUp(self):
        """Test setup - clear global registry."""
        # Clear global registry for testing
        global_registry = get_dataset_registry()
        global_registry.datasets.clear()
        global_registry.factories.clear()
    
    def test_register_dataset_global(self):
        """Test global register_dataset function."""
        class TestDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        register_dataset(name="global_test_dataset", dataset_class=TestDataset)
        
        dataset_names = list_datasets()
        self.assertIn("global_test_dataset", dataset_names)
        
        dataset_info = get_dataset_info("global_test_dataset")
        self.assertIsNotNone(dataset_info)
        self.assertEqual(dataset_info.name, "global_test_dataset")
    
    def test_unregister_dataset_global(self):
        """Test global unregister_dataset function."""
        class TestDataset(Dataset):
            def __init__(self):
                super().__init__()
                self.data = [1, 2, 3]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        register_dataset(name="global_test_dataset", dataset_class=TestDataset)
        self.assertIn("global_test_dataset", list_datasets())
        
        unregister_dataset("global_test_dataset")
        self.assertNotIn("global_test_dataset", list_datasets())
    
    def test_create_dataset_global(self):
        """Test global create_dataset function."""
        class TestDataset(Dataset):
            def __init__(self, size=100):
                super().__init__()
                self.size = size
                self.data = list(range(size))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        register_dataset(name="global_test_dataset", dataset_class=TestDataset)
        
        dataset = create_dataset("global_test_dataset", size=150)
        self.assertIsInstance(dataset, TestDataset)
        self.assertEqual(dataset.size, 150)
        self.assertEqual(len(dataset), 150)
    
    def test_create_dataset_with_config_global(self):
        """Test global create_dataset_with_config function."""
        class TestDataset(Dataset):
            def __init__(self, config=None):
                super().__init__()
                if config:
                    self.size = config.size
                else:
                    self.size = 100
                self.data = list(range(self.size))
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        class TestConfig:
            def __init__(self, size=100):
                self.size = size
        
        register_dataset(name="global_config_dataset", dataset_class=TestDataset)
        
        config = TestConfig(size=200)
        dataset = create_dataset_with_config("global_config_dataset", config)
        self.assertIsInstance(dataset, TestDataset)
        self.assertEqual(dataset.size, 200)
        self.assertEqual(len(dataset), 200)
    
    def test_get_dataset_registry_global(self):
        """Test global get_dataset_registry function."""
        registry = get_dataset_registry()
        self.assertIsInstance(registry, DatasetRegistry)
        
        # Test that it's the same instance
        registry2 = get_dataset_registry()
        self.assertIs(registry, registry2)


if __name__ == '__main__':
    unittest.main()