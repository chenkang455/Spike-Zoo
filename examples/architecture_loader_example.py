#!/usr/bin/env python3
"""
Example usage of the architecture loader with plugin support.
"""

import sys
import os
from pathlib import Path

# Add the spikezoo package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spikezoo.models.architecture_loader import (
    ArchitectureLoader,
    load_architecture_class,
    create_architecture,
    list_available_architectures,
    add_architecture_search_path
)


def example_basic_usage():
    """Demonstrate basic usage of the architecture loader."""
    print("=== Basic Architecture Loader Usage ===\n")
    
    # List available architectures
    print("1. Available built-in architectures:")
    architectures = list_available_architectures()
    for arch in architectures:
        print(f"   - {arch}")
    print()
    
    # Load an architecture class
    print("2. Loading BaseNet architecture class:")
    base_net_class = load_architecture_class("base", "BaseNet", "nets")
    if base_net_class:
        print(f"   Successfully loaded: {base_net_class.__name__}")
        
        # Create an instance
        print("3. Creating BaseNet instance:")
        base_net = base_net_class()
        print(f"   Created instance: {type(base_net).__name__}")
    else:
        print("   Failed to load BaseNet class")
    print()


def example_custom_architecture():
    """Demonstrate loading custom architectures."""
    print("=== Custom Architecture Loading ===\n")
    
    # Add a custom search path
    custom_path = Path(__file__).parent / "custom_archs"
    custom_path.mkdir(exist_ok=True)
    add_architecture_search_path(custom_path)
    
    print(f"1. Added custom search path: {custom_path}")
    
    # Try to load a custom architecture (this would fail in this example
    # since we don't actually create the files, but shows the concept)
    print("2. Attempting to load custom architecture:")
    custom_arch_class = load_architecture_class("custom_model", "CustomNet", "nets")
    if custom_arch_class:
        print(f"   Loaded custom architecture: {custom_arch_class.__name__}")
    else:
        print("   Custom architecture not found (expected in this example)")
    print()


def example_advanced_loader():
    """Demonstrate advanced usage of the ArchitectureLoader class."""
    print("=== Advanced ArchitectureLoader Usage ===\n")
    
    # Create a custom loader instance
    loader = ArchitectureLoader()
    
    print("1. Created custom ArchitectureLoader instance")
    
    # Add a custom search path
    custom_path = Path(__file__).parent / "plugin_archs"
    custom_path.mkdir(exist_ok=True)
    loader.add_search_path(custom_path)
    
    print(f"2. Added plugin search path: {custom_path}")
    
    # List available architectures with custom loader
    print("3. Available architectures with custom loader:")
    architectures = loader.list_available_architectures()
    for arch in architectures:
        print(f"   - {arch}")
    
    # Clear cache
    loader.clear_cache()
    print("4. Cleared loader cache")
    print()


def example_direct_module_loading():
    """Demonstrate direct module loading."""
    print("=== Direct Module Loading ===\n")
    
    # This would work if we had a module in the Python path
    # For demonstration purposes, we'll show the concept
    print("Attempting to load architecture from direct module:")
    
    # This would typically be used for third-party modules
    # arch_class = load_architecture_class("third_party_model", "ThirdPartyNet", "third_party_module")
    # if arch_class:
    #     print(f"Loaded third-party architecture: {arch_class.__name__}")
    
    print("Direct module loading concept demonstrated (actual loading would depend on module existence)")
    print()


def main():
    """Run all examples."""
    print("Architecture Loader Examples\n")
    print("=" * 50)
    
    example_basic_usage()
    example_custom_architecture()
    example_advanced_loader()
    example_direct_module_loading()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()