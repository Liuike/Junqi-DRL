"""
Random agent for Junqi.
Selects actions uniformly at random from legal actions.
"""
import random
from typing import List, Optional
from junqi_drl.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that plays randomly (inherits from BaseAgent)."""
    
    def __init__(self, player_id: int, device: str = "cpu"):
        """
        Initialize the random agent.
        
        Args:
            player_id: The player ID (0 or 1) this agent controls
            device: Device (unused, for BaseAgent compatibility)
        """
        super().__init__(player_id, device)
    
    def choose_action(self, state, legal_actions: Optional[List[int]] = None, eval_mode: bool = False) -> int:
        """
        Select a random legal action (implements BaseAgent.choose_action).
        
        Args:
            state: The current game state
            legal_actions: List of legal actions (computed from state if None)
            eval_mode: Ignored (random agent always random)
            
        Returns:
            A random legal action
        """
        if legal_actions is None:
            legal_actions = state.legal_actions(self.player_id)
        
        if not legal_actions:
            raise ValueError("No legal actions available")
        
        return random.choice(legal_actions)
    
    def step(self, state):
        """
        Deprecated: Use choose_action() instead.
        Provided for backward compatibility.
        
        Args:
            state: The current game state
            
        Returns:
            A random legal action
        """
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)
    
    def reset(self):
        """Reset the agent (no-op for random agent, implements BaseAgent.reset)."""
        pass
    
    def save(self, path: str):
        """
        Save agent (no-op for random agent, implements BaseAgent.save).
        
        Args:
            path: File path (ignored)
        """
        pass
    
    def load(self, path: str):
        """
        Load agent (no-op for random agent, implements BaseAgent.load).
        
        Args:
            path: File path (ignored)
        """
        pass
