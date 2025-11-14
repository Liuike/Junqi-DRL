"""
Random agent for Junqi.
Selects actions uniformly at random from legal actions.
"""
import random


class RandomAgent:
    """Agent that plays randomly."""
    
    def __init__(self, player_id):
        """
        Initialize the random agent.
        
        Args:
            player_id: The player ID (0 or 1) this agent controls
        """
        self.player_id = player_id
    
    def step(self, state):
        """
        Select an action for the current state.
        
        Args:
            state: The current game state
            
        Returns:
            action: A random legal action
        """
        legal_actions = state.legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)
    
    def reset(self):
        """Reset the agent (no-op for random agent)."""
        pass
