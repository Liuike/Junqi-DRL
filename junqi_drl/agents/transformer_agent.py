import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional
from junqi_drl.agents.junqi_transformer import JunqiMoveTransformer
from junqi_drl.agents.base_agent import BaseAgent

class TransformerAgent(BaseAgent):
    """
    Unified Transformer agent for both training (PPO) and evaluation.
    Includes stateful logic to adapt 1-step Transformer output to 2-step Environment input.
    """
    def __init__(
        self,
        player_id: int,
        board_variant: str = "standard",
        model_path: str = None,
        device: str = "cpu",
        deterministic: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        game: Optional[object] = None,
        lr_start: float = 1e-4,
        lr_end: float = 5e-6,
        ent_coef_start: float = 0.02,
        ent_coef_end: float = 0.001,
        training_mode: bool = False
    ):
        """
        Initialize TransformerAgent.
        
        Args:
            player_id: Player ID (0 or 1)
            board_variant: "small" or "standard"
            model_path: Path to load pretrained model (optional)
            device: "cpu" or "cuda"
            deterministic: Use greedy action selection (for eval)
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            game: PySpiel game object (required for training mode)
            lr_start: Starting learning rate (training only)
            lr_end: Ending learning rate (training only)
            ent_coef_start: Starting entropy coefficient (training only)
            ent_coef_end: Ending entropy coefficient (training only)
            training_mode: If True, initialize optimizer and schedulers for training
        """
        super().__init__(player_id, device)
        self.deterministic = deterministic
        self.training_mode = training_mode
        self.game = game
        
        # Model Init
        self.model = JunqiMoveTransformer(
            board_variant=board_variant,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        if training_mode:
            self.model.train()
        else:
            self.model.eval()

        if model_path:
            self.load(model_path)
        
        self.height = self.model.height
        self.width = self.model.width
        self.channels = self.model.channels
        self.seq_len = self.model.seq_len
        self.num_cells = self.seq_len
        self.total_actions = self.model.total_actions
        
        # Internal memory for the 2-step move process
        self.pending_to_idx = None
        
        # Training-specific attributes
        if training_mode:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr_start, eps=1e-5)
            self.lr_scheduler = lambda frac: lr_start + frac * (lr_end - lr_start)
            self.ent_scheduler = lambda frac: ent_coef_start + frac * (ent_coef_end - ent_coef_start) 

    def load(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def process_obs(self, state, player_id=None) -> torch.Tensor:
        """Process observation for both training and evaluation."""
        if player_id is None:
            player_id = self.player_id
        obs_flat = state.observation_tensor(player_id)
        obs_3d = np.array(obs_flat, dtype=np.float32).reshape(self.height, self.width, self.channels)
        return torch.tensor(obs_3d, device=self.device).unsqueeze(0)

    def process_mask(self, state, player_id=None) -> torch.Tensor:
        """Construct legality mask for both training and evaluation."""
        if player_id is None:
            player_id = self.player_id
            
        num_cells = self.seq_len
        full_mask = np.zeros(self.total_actions, dtype=np.float32)
        
        env_legal_actions = state.legal_actions(player_id)
        decode_action = state.decode_action
        
        # Move check - mark all legal from-to pairs
        for from_idx in env_legal_actions:
            if from_idx >= num_cells:
                continue
            row, col = decode_action[from_idx]
            try:
                destinations = state._get_legal_destinations([row, col], player_id)
            except AttributeError:
                destinations = []

            for r_to, c_to in destinations:
                to_idx = r_to * self.width + c_to
                flat_action_idx = from_idx * num_cells + to_idx
                full_mask[flat_action_idx] = 1.0

        return torch.tensor(full_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def update_parameters(self, frac: float):
        """Update learning rate and entropy coefficient based on training fraction."""
        if not self.training_mode:
            raise RuntimeError("update_parameters called but agent not in training mode")
        new_lr = self.lr_scheduler(frac)
        new_ent = self.ent_scheduler(frac)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr, new_ent

    def reset(self):
        """Reset internal state (implements BaseAgent.reset)."""
        self.pending_to_idx = None

    def choose_action(self, state, legal_actions: Optional[List[int]] = None, eval_mode: bool = False) -> int:
        """
        Stateful action selection (implements BaseAgent.choose_action).
        If pending_to_idx is set, return it (Step 2).
        Otherwise, run model, return From, and save To (Step 1).
        
        Args:
            state: Game state
            legal_actions: List of legal actions (computed from state if None)
            eval_mode: If True, use deterministic action selection (ignored, uses self.deterministic)
            
        Returns:
            Action index
        """
        # Note: legal_actions parameter provided for BaseAgent interface compatibility,
        # but we compute them from state for mask generation
        
        # Case 1: executing the second half of a move
        if self.pending_to_idx is not None:
            if state.selecting_piece:
                # Synchronization error check
                # print("Warning: Pending move exists but Env expects Select. Clearing pending.")
                self.pending_to_idx = None
            else:
                action = self.pending_to_idx
                self.pending_to_idx = None
                return action

        # Case 2: Start new move (Step 1)
        legal_actions = state.legal_actions(self.player_id)
        
        # If no legal actions, game should be terminal - shouldn't reach here
        if not legal_actions:
            raise RuntimeError(f"Player {self.player_id} has no legal actions but game is not terminal")
        
        # Get model prediction
        obs_tensor = self.process_obs(state)
        mask_tensor = self.process_mask(state)

        with torch.no_grad():
            action_logits, _ = self.model(obs_tensor, mask=mask_tensor)
            action_logits = action_logits.squeeze(0)

        if self.deterministic:
            action_int = torch.argmax(action_logits).item()
        else:
            probs = torch.softmax(action_logits, dim=0)
            action_int = torch.multinomial(probs, 1).item()

        # Decode from flattened action to from_idx and to_idx
        num_cells = self.seq_len
        from_idx = action_int // num_cells
        to_idx = action_int % num_cells

        # Check if the model's chosen 'from_idx' is actually legal.
        if from_idx not in legal_actions:
            fallback_action = np.random.choice(legal_actions)
            return fallback_action.item()

        # Only set pending if the first step is valid
        self.pending_to_idx = to_idx
        return from_idx

    def save(self, path: str):
        """
        Save model checkpoint (implements BaseAgent.save).
        
        Args:
            path: File path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'player_id': self.player_id,
            'deterministic': self.deterministic,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """
        Load model checkpoint (implements BaseAgent.load).
        
        Args:
            path: File path to load checkpoint from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
