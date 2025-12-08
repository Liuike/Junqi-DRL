"""Abstract base class for all Junqi agents."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseAgent(ABC):
    """
    Abstract base class defining the interface for all Junqi agents.
    
    All agent implementations (DRQN, Transformer, Random, etc.) must inherit
    from this class and implement its abstract methods.
    """
    
    def __init__(self, player_id: int, device: str = "cpu"):
        """
        Initialize the agent.
        
        Args:
            player_id: Player ID (0 or 1)
            device: Device for PyTorch tensors ("cpu" or "cuda")
        """
        self.player_id = player_id
        self.device = device
    
    @abstractmethod
    def choose_action(
        self, 
        state, 
        legal_actions: Optional[List[int]] = None,
        eval_mode: bool = False
    ) -> int:
        """
        Select an action given the current game state.
        
        Args:
            state: OpenSpiel game state
            legal_actions: List of legal action indices (optional, will be computed if None)
            eval_mode: If True, use deterministic/greedy action selection
            
        Returns:
            Action index to take
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset agent's internal state.
        
        This should be called at the start of each episode to reset
        any recurrent hidden states, pending actions, or other stateful components.
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save agent's model/parameters to disk.
        
        Args:
            path: File path to save to
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load agent's model/parameters from disk.
        
        Args:
            path: File path to load from
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return agent configuration as a dictionary.
        
        Useful for logging and reproducibility.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {
            "player_id": self.player_id,
            "device": self.device,
            "agent_type": self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(player_id={self.player_id}, device={self.device})"
