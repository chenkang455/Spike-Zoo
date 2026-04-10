#!/usr/bin/env python3
"""
Example of using the SpikeZoo visualization system.
"""

import sys
import os
import numpy as np
import time

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.utils.visualization_utils import (
    VisualizationConfig,
    VisualizationManager,
    get_visualization_manager,
    setup_visualization,
    log_scalar,
    log_image,
    log_histogram,
    log_text,
    log_config,
    flush_visualization,
    close_visualization
)


def example_basic_visualization():
    """Example of basic visualization usage."""
    print("=== Basic Visualization Example ===\n")
    
    # Create visualization configuration
    config = VisualizationConfig(
        experiment_name="basic_example",
        log_dir="./logs",
        tensorboard_enabled=True,
        wandb_enabled=False,  # Disable WandB for this example
        plot_enabled=True
    )
    
    # Create visualization manager
    vis_manager = VisualizationManager(config)
    
    # Log some scalars
    for step in range(100):
        # Simulate training metrics
        loss = 1.0 * np.exp(-step * 0.1) + np.random.normal(0, 0.05)
        accuracy = 0.5 * (1 - np.exp(-step * 0.05)) + np.random.normal(0, 0.02)
        
        vis_manager.log_scalar("train/loss", loss, step)
        vis_manager.log_scalar("train/accuracy", accuracy, step)
        
        # Log some histograms periodically
        if step % 20 == 0:
            # Simulate weight distributions
            weights = np.random.normal(0, 1, 1000) * np.exp(-step * 0.01)
            vis_manager.log_histogram("weights/layer1", weights, step)
    
    # Log configuration
    config_dict = {
        "model": "ExampleModel",
        "dataset": "ExampleDataset",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    vis_manager.log_config(config_dict)
    
    # Log some text
    vis_manager.log_text("notes", "This is a basic visualization example", 0)
    
    # Flush logs
    vis_manager.flush()
    
    # Close visualization manager
    vis_manager.close()
    
    print("Basic visualization example completed")
    print()


def example_global_visualization():
    """Example of global visualization usage."""
    print("=== Global Visualization Example ===\n")
    
    # Setup global visualization
    config = VisualizationConfig(
        experiment_name="global_example",
        log_dir="./logs",
        tensorboard_enabled=True,
        wandb_enabled=False,
        plot_enabled=True
    )
    
    setup_visualization(config)
    
    # Use convenience functions
    log_scalar("global/loss", 0.5, 1)
    log_scalar("global/accuracy", 0.8, 1)
    
    # Log configuration
    log_config({
        "example": "global visualization",
        "timestamp": time.time()
    })
    
    # Log text
    log_text("global/notes", "This is a global visualization example", 1)
    
    # Flush logs
    flush_visualization()
    
    # Close visualization
    close_visualization()
    
    print("Global visualization example completed")
    print()


def example_image_logging():
    """Example of image logging."""
    print("=== Image Logging Example ===\n")
    
    # Create visualization manager
    config = VisualizationConfig(
        experiment_name="image_example",
        log_dir="./logs",
        tensorboard_enabled=True,
        wandb_enabled=False,
        plot_enabled=True
    )
    
    vis_manager = VisualizationManager(config)
    
    # Log different types of images
    # Grayscale image
    grayscale_img = np.random.rand(64, 64)
    vis_manager.log_image("images/grayscale", grayscale_img, 0)
    
    # RGB image
    rgb_img = np.random.rand(32, 32, 3)
    vis_manager.log_image("images/rgb", rgb_img, 0)
    
    # RGBA image
    rgba_img = np.random.rand(16, 16, 4)
    vis_manager.log_image("images/rgba", rgba_img, 0)
    
    # Log multiple images with different steps
    for step in range(5):
        # Simulate evolving image
        img = np.sin(np.linspace(0, 2*np.pi, 32))[:, None] * np.cos(np.linspace(0, 2*np.pi, 32))[None, :] + step * 0.1
        img = np.stack([img, img * 0.5, img * 0.2], axis=-1)  # RGB
        vis_manager.log_image("images/evolving", img, step)
    
    # Flush and close
    vis_manager.flush()
    vis_manager.close()
    
    print("Image logging example completed")
    print()


def example_mixed_logging():
    """Example of mixed logging (scalars, images, histograms, text)."""
    print("=== Mixed Logging Example ===\n")
    
    # Create visualization manager
    config = VisualizationConfig(
        experiment_name="mixed_example",
        log_dir="./logs",
        tensorboard_enabled=True,
        wandb_enabled=False,
        plot_enabled=True
    )
    
    vis_manager = VisualizationManager(config)
    
    # Simulate training loop
    for epoch in range(10):
        # Simulate training metrics
        train_loss = 1.0 * np.exp(-epoch * 0.3) + np.random.normal(0, 0.1)
        val_loss = 1.0 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.1)
        train_acc = 0.7 + 0.25 * (1 - np.exp(-epoch * 0.5)) + np.random.normal(0, 0.05)
        val_acc = 0.65 + 0.3 * (1 - np.exp(-epoch * 0.4)) + np.random.normal(0, 0.05)
        
        # Log scalars
        vis_manager.log_scalar("loss/train", train_loss, epoch)
        vis_manager.log_scalar("loss/val", val_loss, epoch)
        vis_manager.log_scalar("accuracy/train", train_acc, epoch)
        vis_manager.log_scalar("accuracy/val", val_acc, epoch)
        
        # Log histograms of simulated gradients
        if epoch % 3 == 0:
            gradients = np.random.normal(0, 0.1 * np.exp(-epoch * 0.1), 1000)
            vis_manager.log_histogram("gradients/layer1", gradients, epoch)
            
            weights = np.random.normal(0, 1.0 * np.exp(-epoch * 0.05), 1000)
            vis_manager.log_histogram("weights/layer1", weights, epoch)
        
        # Log sample images
        if epoch % 5 == 0:
            # Simulate input image
            input_img = np.random.rand(28, 28)
            vis_manager.log_image("samples/input", input_img, epoch)
            
            # Simulate output image
            output_img = np.random.rand(28, 28)
            vis_manager.log_image("samples/output", output_img, epoch)
            
            # Simulate difference image
            diff_img = np.abs(input_img - output_img)
            vis_manager.log_image("samples/difference", diff_img, epoch)
        
        # Log training notes
        if epoch == 0:
            vis_manager.log_text("notes/training_setup", f"Training started with config: epochs={10}, batch_size=32", epoch)
        elif epoch == 5:
            vis_manager.log_text("notes/midpoint", f"Halfway through training. Current loss: {train_loss:.4f}", epoch)
        elif epoch == 9:
            vis_manager.log_text("notes/training_complete", f"Training completed. Final accuracy: {val_acc:.4f}", epoch)
    
    # Log final configuration
    final_config = {
        "model_architecture": "CNN",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "final_val_accuracy": val_acc
    }
    vis_manager.log_config(final_config)
    
    # Flush and close
    vis_manager.flush()
    vis_manager.close()
    
    print("Mixed logging example completed")
    print()


def example_disabled_visualization():
    """Example of visualization with disabled backends."""
    print("=== Disabled Visualization Example ===\n")
    
    # Create configuration with all backends disabled
    config = VisualizationConfig(
        enabled=False,  # Completely disable visualization
        experiment_name="disabled_example",
        log_dir="./logs",
        tensorboard_enabled=False,
        wandb_enabled=False,
        plot_enabled=False
    )
    
    # Create visualization manager
    vis_manager = VisualizationManager(config)
    
    # These calls should not raise exceptions even with visualization disabled
    vis_manager.log_scalar("test/loss", 0.5, 1)
    vis_manager.log_image("test/image", np.random.rand(10, 10), 1)
    vis_manager.log_histogram("test/hist", np.random.rand(100), 1)
    vis_manager.log_text("test/text", "This should not be logged", 1)
    vis_manager.log_config({"test": "config"})
    
    # Flush and close
    vis_manager.flush()
    vis_manager.close()
    
    print("Disabled visualization example completed")
    print()


if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs("./logs", exist_ok=True)
    
    example_basic_visualization()
    example_global_visualization()
    example_image_logging()
    example_mixed_logging()
    example_disabled_visualization()
    
    print("All visualization examples completed!")
    print("Check the logs directory for visualization outputs.")