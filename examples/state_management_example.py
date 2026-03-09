#!/usr/bin/env python3
"""
Example of using the SpikeZoo pipeline state management system.
"""

import sys
import os

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.pipeline import (
    StateManager, 
    PipelineMode, 
    PipelineState,
    create_state_manager,
    get_allowed_modes,
    get_allowed_states
)


def example_state_manager_basic():
    """Example of basic state manager usage."""
    print("=== SpikeZoo State Management Example ===\n")
    
    # Create state manager
    state_manager = create_state_manager(PipelineMode.SINGLE_MODE)
    
    print("1. Initial state:")
    info = state_manager.get_state_info()
    print(f"   Mode: {info['mode']}")
    print(f"   State: {info['state']}")
    print()
    
    # Change mode
    print("2. Changing mode:")
    state_manager.transition_to_mode(PipelineMode.TRAIN_MODE)
    info = state_manager.get_state_info()
    print(f"   New mode: {info['mode']}")
    print(f"   Previous mode: {info['previous_mode']}")
    print()
    
    # Change state
    print("3. Changing state:")
    state_manager.transition_to_state(PipelineState.TRAINING)
    info = state_manager.get_state_info()
    print(f"   New state: {info['state']}")
    print(f"   Previous state: {info['previous_state']}")
    print()
    
    # Check state
    print("4. State checks:")
    print(f"   Is training mode: {state_manager.is_training_mode()}")
    print(f"   Is inference mode: {state_manager.is_inference_mode()}")
    print(f"   Is running: {state_manager.is_running()}")
    print(f"   Is ready: {state_manager.is_ready()}")
    print()


def example_state_transitions():
    """Example of state transitions."""
    print("=== State Transitions Example ===\n")
    
    # Create state manager in train mode
    state_manager = create_state_manager(PipelineMode.TRAIN_MODE)
    state_manager.transition_to_state(PipelineState.READY)
    
    print("1. Valid transitions:")
    # Test valid transitions
    valid_transitions = [
        (PipelineState.RUNNING, "Running"),
        (PipelineState.TRAINING, "Training"),
        (PipelineState.STOPPING, "Stopping")
    ]
    
    for state, description in valid_transitions:
        can_transition = state_manager.can_transition_to_state(state)
        print(f"   To {description}: {'Allowed' if can_transition else 'Denied'}")
    print()
    
    # Test actual transition
    print("2. Performing transition:")
    success = state_manager.transition_to_state(PipelineState.TRAINING)
    print(f"   Transition to TRAINING: {'Success' if success else 'Failed'}")
    print(f"   Current state: {state_manager.state.value}")
    print()
    
    # Test invalid transition
    print("3. Invalid transition attempt:")
    success = state_manager.transition_to_state(PipelineState.STOPPED)
    print(f"   Transition to STOPPED: {'Success' if success else 'Failed'}")
    print(f"   Current state: {state_manager.state.value}")
    print()


def example_state_data_storage():
    """Example of state data storage."""
    print("=== State Data Storage Example ===\n")
    
    state_manager = create_state_manager(PipelineMode.SINGLE_MODE)
    
    # Store data
    print("1. Storing state data:")
    state_manager.store_state_data("epoch", 5)
    state_manager.store_state_data("loss", 0.023)
    state_manager.store_state_data("accuracy", 0.98)
    
    # Retrieve data
    epoch = state_manager.get_state_data("epoch")
    loss = state_manager.get_state_data("loss")
    accuracy = state_manager.get_state_data("accuracy")
    
    print(f"   Epoch: {epoch}")
    print(f"   Loss: {loss}")
    print(f"   Accuracy: {accuracy}")
    print()
    
    # Retrieve non-existent data with default
    metrics = state_manager.get_state_data("metrics", {})
    print(f"   Non-existent data with default: {metrics}")
    print()
    
    # Clear data
    print("2. Clearing state data:")
    state_manager.clear_state_data("loss")
    loss = state_manager.get_state_data("loss")
    print(f"   Loss after clearing: {loss}")
    
    state_manager.clear_state_data()
    epoch = state_manager.get_state_data("epoch")
    accuracy = state_manager.get_state_data("accuracy")
    print(f"   Epoch after clearing all: {epoch}")
    print(f"   Accuracy after clearing all: {accuracy}")
    print()


def example_state_history():
    """Example of state history tracking."""
    print("=== State History Tracking Example ===\n")
    
    state_manager = create_state_manager(PipelineMode.SINGLE_MODE)
    
    # Perform several state transitions
    transitions = [
        (PipelineState.READY, "Ready"),
        (PipelineState.RUNNING, "Running"),
        (PipelineState.STOPPING, "Stopping"),
        (PipelineState.STOPPED, "Stopped"),
        (PipelineState.READY, "Ready again")
    ]
    
    print("1. Performing state transitions:")
    for state, description in transitions:
        success = state_manager.transition_to_state(state)
        print(f"   {description}: {'Success' if success else 'Failed'}")
    print()
    
    # View state history
    print("2. State history:")
    history = state_manager.get_state_info()['state_history']
    for i, transition in enumerate(history):
        print(f"   {i+1}. {transition['from']} -> {transition['to']} (mode: {transition['mode']})")
    print()


def example_mode_transitions():
    """Example of mode transitions."""
    print("=== Mode Transitions Example ===\n")
    
    state_manager = create_state_manager(PipelineMode.SINGLE_MODE)
    
    print("1. Allowed modes:")
    modes = get_allowed_modes()
    for mode in modes:
        print(f"   - {mode}")
    print()
    
    print("2. Mode transitions:")
    mode_transitions = [
        (PipelineMode.TRAIN_MODE, "Train Mode"),
        (PipelineMode.MULTI_MODE, "Multi Mode"),
        (PipelineMode.SINGLE_MODE, "Single Mode")
    ]
    
    for mode, description in mode_transitions:
        success = state_manager.transition_to_mode(mode)
        print(f"   To {description}: {'Changed' if success else 'No change'}")
        print(f"     Current mode: {state_manager.mode.value}")
    print()


def example_state_queries():
    """Example of state queries."""
    print("=== State Queries Example ===\n")
    
    # Test different modes and states
    configs = [
        (PipelineMode.SINGLE_MODE, PipelineState.READY, "Single Mode Ready"),
        (PipelineMode.TRAIN_MODE, PipelineState.TRAINING, "Train Mode Training"),
        (PipelineMode.MULTI_MODE, PipelineState.INFERRING, "Multi Mode Inferring")
    ]
    
    for mode, state, description in configs:
        print(f"1. {description}:")
        state_manager = create_state_manager(mode)
        state_manager.transition_to_state(state)
        
        print(f"   Mode: {state_manager.mode.value}")
        print(f"   State: {state_manager.state.value}")
        print(f"   Is training mode: {state_manager.is_training_mode()}")
        print(f"   Is inference mode: {state_manager.is_inference_mode()}")
        print(f"   Is running: {state_manager.is_running()}")
        print(f"   Is ready: {state_manager.is_ready()}")
        print(f"   Has error: {state_manager.has_error()}")
        print()


if __name__ == "__main__":
    example_state_manager_basic()
    example_state_transitions()
    example_state_data_storage()
    example_state_history()
    example_mode_transitions()
    example_state_queries()