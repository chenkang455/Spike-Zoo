from enum import Enum
from typing import Optional, Dict, Any
import logging


class PipelineMode(Enum):
    """Pipeline operation modes."""
    SINGLE_MODE = "single_mode"      # Single model inference
    MULTI_MODE = "multi_mode"        # Multiple model ensemble inference
    TRAIN_MODE = "train_mode"        # Model training


class PipelineState(Enum):
    """Pipeline states."""
    INITIALIZING = "initializing"     # Pipeline is being initialized
    READY = "ready"                  # Pipeline is ready for operations
    RUNNING = "running"              # Pipeline is currently executing
    TRAINING = "training"            # Pipeline is in training mode
    INFERRING = "inferring"          # Pipeline is in inference mode
    STOPPING = "stopping"            # Pipeline is being stopped
    STOPPED = "stopped"              # Pipeline is stopped
    ERROR = "error"                  # Pipeline encountered an error


class StateManager:
    """Manages pipeline state and mode transitions."""
    
    def __init__(self, initial_mode: PipelineMode = PipelineMode.SINGLE_MODE):
        """
        Initialize state manager.
        
        Args:
            initial_mode: Initial pipeline mode
        """
        self.mode = initial_mode
        self.state = PipelineState.INITIALIZING
        self.previous_state = None
        self.previous_mode = None
        self.logger = logging.getLogger(__name__)
        self.state_history = []
        self.max_history = 10  # Keep last 10 state transitions
        
        # Store additional state data
        self.state_data: Dict[str, Any] = {}
    
    def set_mode(self, mode: PipelineMode) -> bool:
        """
        Set pipeline mode.
        
        Args:
            mode: New pipeline mode
            
        Returns:
            True if mode changed, False otherwise
        """
        if self.mode == mode:
            return False
        
        self.logger.debug(f"Changing mode from {self.mode.value} to {mode.value}")
        self.previous_mode = self.mode
        self.mode = mode
        return True
    
    def set_state(self, state: PipelineState) -> bool:
        """
        Set pipeline state.
        
        Args:
            state: New pipeline state
            
        Returns:
            True if state changed, False otherwise
        """
        if self.state == state:
            return False
        
        # Record state transition
        self.state_history.append({
            'from': self.state.value,
            'to': state.value,
            'mode': self.mode.value
        })
        
        # Keep only last N transitions
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        self.logger.debug(f"Changing state from {self.state.value} to {state.value}")
        self.previous_state = self.state
        self.state = state
        return True
    
    def can_transition_to_mode(self, new_mode: PipelineMode) -> bool:
        """
        Check if pipeline can transition to a new mode.
        
        Args:
            new_mode: Target mode
            
        Returns:
            True if transition is allowed, False otherwise
        """
        # Some transitions might be restricted based on current state
        # For now, we allow all mode transitions
        return True
    
    def can_transition_to_state(self, new_state: PipelineState) -> bool:
        """
        Check if pipeline can transition to a new state.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition is allowed, False otherwise
        """
        # Define valid state transitions
        valid_transitions = {
            PipelineState.INITIALIZING: [
                PipelineState.READY, 
                PipelineState.ERROR
            ],
            PipelineState.READY: [
                PipelineState.RUNNING, 
                PipelineState.TRAINING, 
                PipelineState.INFERRING,
                PipelineState.STOPPING,
                PipelineState.ERROR
            ],
            PipelineState.RUNNING: [
                PipelineState.STOPPING,
                PipelineState.ERROR,
                PipelineState.READY
            ],
            PipelineState.TRAINING: [
                PipelineState.STOPPING,
                PipelineState.ERROR,
                PipelineState.READY
            ],
            PipelineState.INFERRING: [
                PipelineState.STOPPING,
                PipelineState.ERROR,
                PipelineState.READY
            ],
            PipelineState.STOPPING: [
                PipelineState.STOPPED,
                PipelineState.ERROR
            ],
            PipelineState.STOPPED: [
                PipelineState.READY,
                PipelineState.ERROR
            ],
            PipelineState.ERROR: [
                PipelineState.STOPPING,
                PipelineState.STOPPED,
                PipelineState.READY
            ]
        }
        
        current_valid_transitions = valid_transitions.get(self.state, [])
        return new_state in current_valid_transitions
    
    def transition_to_mode(self, mode: PipelineMode) -> bool:
        """
        Transition to a new mode if allowed.
        
        Args:
            mode: Target mode
            
        Returns:
            True if transition successful, False otherwise
        """
        if not self.can_transition_to_mode(mode):
            self.logger.warning(f"Mode transition from {self.mode.value} to {mode.value} not allowed")
            return False
        
        return self.set_mode(mode)
    
    def transition_to_state(self, state: PipelineState) -> bool:
        """
        Transition to a new state if allowed.
        
        Args:
            state: Target state
            
        Returns:
            True if transition successful, False otherwise
        """
        if not self.can_transition_to_state(state):
            self.logger.warning(f"State transition from {self.state.value} to {state.value} not allowed")
            return False
        
        return self.set_state(state)
    
    def is_training_mode(self) -> bool:
        """Check if pipeline is in training mode."""
        return self.mode == PipelineMode.TRAIN_MODE
    
    def is_inference_mode(self) -> bool:
        """Check if pipeline is in inference mode."""
        return self.mode in [PipelineMode.SINGLE_MODE, PipelineMode.MULTI_MODE]
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.state in [
            PipelineState.RUNNING, 
            PipelineState.TRAINING, 
            PipelineState.INFERRING
        ]
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready."""
        return self.state == PipelineState.READY
    
    def has_error(self) -> bool:
        """Check if pipeline has error."""
        return self.state == PipelineState.ERROR
    
    def store_state_data(self, key: str, value: Any):
        """
        Store additional state data.
        
        Args:
            key: Data key
            value: Data value
        """
        self.state_data[key] = value
    
    def get_state_data(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get stored state data.
        
        Args:
            key: Data key
            default: Default value if key not found
            
        Returns:
            Stored data or default value
        """
        return self.state_data.get(key, default)
    
    def clear_state_data(self, key: Optional[str] = None):
        """
        Clear state data.
        
        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is None:
            self.state_data.clear()
        else:
            self.state_data.pop(key, None)
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current state information.
        
        Returns:
            State information dictionary
        """
        return {
            'mode': self.mode.value,
            'state': self.state.value,
            'previous_mode': self.previous_mode.value if self.previous_mode else None,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'state_history': self.state_history[-5:] if self.state_history else []  # Last 5 transitions
        }


# Convenience functions
def create_state_manager(initial_mode: PipelineMode = PipelineMode.SINGLE_MODE) -> StateManager:
    """
    Create a new state manager.
    
    Args:
        initial_mode: Initial pipeline mode
        
    Returns:
        StateManager instance
    """
    return StateManager(initial_mode)


def get_allowed_modes() -> list:
    """
    Get list of all allowed pipeline modes.
    
    Returns:
        List of PipelineMode values
    """
    return [mode.value for mode in PipelineMode]


def get_allowed_states() -> list:
    """
    Get list of all allowed pipeline states.
    
    Returns:
        List of PipelineState values
    """
    return [state.value for state in PipelineState]