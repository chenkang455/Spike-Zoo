import unittest
import sys
import os

# Add the spikezoo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spikezoo.pipeline.state_manager import (
    StateManager,
    PipelineMode,
    PipelineState,
    create_state_manager,
    get_allowed_modes,
    get_allowed_states
)


class TestStateManager(unittest.TestCase):
    """StateManager unit tests."""
    
    def test_state_manager_creation(self):
        """Test StateManager creation."""
        # Test with default mode
        state_manager = StateManager()
        self.assertEqual(state_manager.mode, PipelineMode.SINGLE_MODE)
        self.assertEqual(state_manager.state, PipelineState.INITIALIZING)
        self.assertIsNone(state_manager.previous_mode)
        self.assertIsNone(state_manager.previous_state)
        
        # Test with specific mode
        state_manager = StateManager(PipelineMode.TRAIN_MODE)
        self.assertEqual(state_manager.mode, PipelineMode.TRAIN_MODE)
        self.assertEqual(state_manager.state, PipelineState.INITIALIZING)
    
    def test_set_mode(self):
        """Test setting mode."""
        state_manager = StateManager()
        
        # Change mode
        changed = state_manager.set_mode(PipelineMode.TRAIN_MODE)
        self.assertTrue(changed)
        self.assertEqual(state_manager.mode, PipelineMode.TRAIN_MODE)
        self.assertEqual(state_manager.previous_mode, PipelineMode.SINGLE_MODE)
        
        # Set same mode again
        changed = state_manager.set_mode(PipelineMode.TRAIN_MODE)
        self.assertFalse(changed)
    
    def test_set_state(self):
        """Test setting state."""
        state_manager = StateManager()
        
        # Change state
        changed = state_manager.set_state(PipelineState.READY)
        self.assertTrue(changed)
        self.assertEqual(state_manager.state, PipelineState.READY)
        self.assertEqual(state_manager.previous_state, PipelineState.INITIALIZING)
        
        # Set same state again
        changed = state_manager.set_state(PipelineState.READY)
        self.assertFalse(changed)
    
    def test_can_transition_to_mode(self):
        """Test mode transition validation."""
        state_manager = StateManager()
        
        # All mode transitions should be allowed for now
        self.assertTrue(state_manager.can_transition_to_mode(PipelineMode.SINGLE_MODE))
        self.assertTrue(state_manager.can_transition_to_mode(PipelineMode.MULTI_MODE))
        self.assertTrue(state_manager.can_transition_to_mode(PipelineMode.TRAIN_MODE))
    
    def test_can_transition_to_state(self):
        """Test state transition validation."""
        state_manager = StateManager()
        state_manager.set_state(PipelineState.READY)
        
        # Test valid transitions from READY
        self.assertTrue(state_manager.can_transition_to_state(PipelineState.RUNNING))
        self.assertTrue(state_manager.can_transition_to_state(PipelineState.TRAINING))
        self.assertTrue(state_manager.can_transition_to_state(PipelineState.INFERRING))
        self.assertTrue(state_manager.can_transition_to_state(PipelineState.STOPPING))
        self.assertTrue(state_manager.can_transition_to_state(PipelineState.ERROR))
        
        # Test invalid transition
        self.assertFalse(state_manager.can_transition_to_state(PipelineState.STOPPED))
    
    def test_transition_to_mode(self):
        """Test mode transition."""
        state_manager = StateManager()
        
        # Valid transition
        success = state_manager.transition_to_mode(PipelineMode.TRAIN_MODE)
        self.assertTrue(success)
        self.assertEqual(state_manager.mode, PipelineMode.TRAIN_MODE)
        
        # Same mode transition
        success = state_manager.transition_to_mode(PipelineMode.TRAIN_MODE)
        self.assertFalse(success)
    
    def test_transition_to_state(self):
        """Test state transition."""
        state_manager = StateManager()
        state_manager.set_state(PipelineState.READY)
        
        # Valid transition
        success = state_manager.transition_to_state(PipelineState.TRAINING)
        self.assertTrue(success)
        self.assertEqual(state_manager.state, PipelineState.TRAINING)
        
        # Invalid transition
        success = state_manager.transition_to_state(PipelineState.STOPPED)
        self.assertFalse(success)
        # State should remain unchanged
        self.assertEqual(state_manager.state, PipelineState.TRAINING)
    
    def test_mode_checks(self):
        """Test mode check methods."""
        # Test training mode
        state_manager = StateManager(PipelineMode.TRAIN_MODE)
        self.assertTrue(state_manager.is_training_mode())
        self.assertFalse(state_manager.is_inference_mode())
        
        # Test inference modes
        state_manager = StateManager(PipelineMode.SINGLE_MODE)
        self.assertFalse(state_manager.is_training_mode())
        self.assertTrue(state_manager.is_inference_mode())
        
        state_manager = StateManager(PipelineMode.MULTI_MODE)
        self.assertFalse(state_manager.is_training_mode())
        self.assertTrue(state_manager.is_inference_mode())
    
    def test_state_checks(self):
        """Test state check methods."""
        state_manager = StateManager()
        
        # Test is_running
        state_manager.set_state(PipelineState.RUNNING)
        self.assertTrue(state_manager.is_running())
        
        state_manager.set_state(PipelineState.TRAINING)
        self.assertTrue(state_manager.is_running())
        
        state_manager.set_state(PipelineState.INFERRING)
        self.assertTrue(state_manager.is_running())
        
        state_manager.set_state(PipelineState.READY)
        self.assertFalse(state_manager.is_running())
        
        # Test is_ready
        state_manager.set_state(PipelineState.READY)
        self.assertTrue(state_manager.is_ready())
        
        state_manager.set_state(PipelineState.RUNNING)
        self.assertFalse(state_manager.is_ready())
        
        # Test has_error
        state_manager.set_state(PipelineState.ERROR)
        self.assertTrue(state_manager.has_error())
        
        state_manager.set_state(PipelineState.READY)
        self.assertFalse(state_manager.has_error())
    
    def test_state_data_storage(self):
        """Test state data storage methods."""
        state_manager = StateManager()
        
        # Store data
        state_manager.store_state_data("test_key", "test_value")
        state_manager.store_state_data("number", 42)
        state_manager.store_state_data("list", [1, 2, 3])
        
        # Retrieve data
        self.assertEqual(state_manager.get_state_data("test_key"), "test_value")
        self.assertEqual(state_manager.get_state_data("number"), 42)
        self.assertEqual(state_manager.get_state_data("list"), [1, 2, 3])
        
        # Retrieve with default
        self.assertEqual(state_manager.get_state_data("nonexistent", "default"), "default")
        
        # Clear specific data
        state_manager.clear_state_data("number")
        self.assertIsNone(state_manager.get_state_data("number"))
        
        # Clear all data
        state_manager.clear_state_data()
        self.assertIsNone(state_manager.get_state_data("test_key"))
        self.assertIsNone(state_manager.get_state_data("list"))
    
    def test_state_info(self):
        """Test state info method."""
        state_manager = StateManager(PipelineMode.TRAIN_MODE)
        
        # Perform some transitions to build history
        state_manager.set_state(PipelineState.READY)
        state_manager.set_state(PipelineState.TRAINING)
        state_manager.set_state(PipelineState.STOPPING)
        
        # Get state info
        info = state_manager.get_state_info()
        
        self.assertEqual(info['mode'], 'train_mode')
        self.assertEqual(info['state'], 'stopping')
        self.assertEqual(info['previous_mode'], 'train_mode')
        self.assertEqual(info['previous_state'], 'training')
        self.assertIsInstance(info['state_history'], list)
        self.assertLessEqual(len(info['state_history']), 10)  # Max history limit
    
    def test_state_history_limit(self):
        """Test state history limit."""
        state_manager = StateManager()
        
        # Perform more than 10 transitions
        for i in range(15):
            state_manager.set_state(PipelineState.READY if i % 2 == 0 else PipelineState.RUNNING)
        
        # Check history is limited
        info = state_manager.get_state_info()
        self.assertLessEqual(len(info['state_history']), 10)


class TestGlobalFunctions(unittest.TestCase):
    """Test global state management functions."""
    
    def test_create_state_manager(self):
        """Test create_state_manager function."""
        # Test with default mode
        state_manager = create_state_manager()
        self.assertIsInstance(state_manager, StateManager)
        self.assertEqual(state_manager.mode, PipelineMode.SINGLE_MODE)
        
        # Test with specific mode
        state_manager = create_state_manager(PipelineMode.TRAIN_MODE)
        self.assertEqual(state_manager.mode, PipelineMode.TRAIN_MODE)
    
    def test_get_allowed_modes(self):
        """Test get_allowed_modes function."""
        modes = get_allowed_modes()
        self.assertIsInstance(modes, list)
        self.assertIn('single_mode', modes)
        self.assertIn('multi_mode', modes)
        self.assertIn('train_mode', modes)
    
    def test_get_allowed_states(self):
        """Test get_allowed_states function."""
        states = get_allowed_states()
        self.assertIsInstance(states, list)
        self.assertIn('initializing', states)
        self.assertIn('ready', states)
        self.assertIn('running', states)
        self.assertIn('training', states)
        self.assertIn('inferring', states)
        self.assertIn('stopping', states)
        self.assertIn('stopped', states)
        self.assertIn('error', states)


if __name__ == '__main__':
    unittest.main()