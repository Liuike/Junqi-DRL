import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import List, Optional, Dict, Any
from collections import deque

from .base_agent import BaseAgent
from ..core.replay_buffer import ReplayBuffer, TemporalStratifiedReplayBuffer
from ..utils.observation import get_board_config, process_observation

class DRQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Deeper network for better representation
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.q_head = nn.Linear(hidden_size, action_dim)
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, obs_seq, hidden=None):
        # obs_seq: (batch, seq_len, obs_dim)
        x = self.relu(self.fc1(obs_seq))
        x = self.relu(self.fc2(x))
        x = self.layer_norm(x)
        
        gru_out, hidden = self.gru(x, hidden)
        
        x = self.relu(self.fc3(gru_out))
        q_values = self.q_head(x)
        return q_values, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class SpatialDRQNetwork(nn.Module):
    """
    Spatial-aware DRQN using CNN + GRU.
    Processes (H, W, C) observations with convolutional layers before recurrent processing.
    """
    def __init__(self, board_config: Dict[str, int], action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.height = board_config['height']
        self.width = board_config['width']
        self.channels = board_config['channels']
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # Spatial feature extractor (CNN)
        # Input: (B, C, H, W) - PyTorch conv expects channels-first
        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Calculate flattened size after convs (no pooling, same spatial dims)
        self.conv_output_size = self.height * self.width * 128
        
        # Recurrent processing
        self.fc_pre_gru = nn.Linear(self.conv_output_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        # Q-value head
        self.fc_post_gru = nn.Linear(hidden_size, hidden_size)
        self.q_head = nn.Linear(hidden_size, action_dim)
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, obs_seq, hidden=None):
        """
        Args:
            obs_seq: (B, T, H, W, C) or (B, 1, H, W, C)
            hidden: GRU hidden state
            
        Returns:
            q_values: (B, T, action_dim)
            hidden: Updated GRU hidden state
        """
        batch_size, seq_len = obs_seq.shape[0], obs_seq.shape[1]
        
        # Reshape for CNN: (B*T, C, H, W)
        obs_flat = obs_seq.reshape(batch_size * seq_len, self.height, self.width, self.channels)
        obs_flat = obs_flat.permute(0, 3, 1, 2)  # (B*T, C, H, W)
        
        # Convolutional feature extraction
        x = self.relu(self.conv1(obs_flat))  # (B*T, 64, H, W)
        x = self.relu(self.conv2(x))         # (B*T, 128, H, W)
        x = self.relu(self.conv3(x))         # (B*T, 128, H, W)
        
        # Flatten spatial dims
        x = x.reshape(batch_size * seq_len, -1)  # (B*T, conv_output_size)
        
        # Pre-GRU processing
        x = self.relu(self.fc_pre_gru(x))  # (B*T, hidden_size)
        x = self.layer_norm(x)
        
        # Reshape back for GRU: (B, T, hidden_size)
        x = x.reshape(batch_size, seq_len, self.hidden_size)
        
        # Recurrent processing
        gru_out, hidden = self.gru(x, hidden)  # (B, T, hidden_size)
        
        # Q-value head
        x = self.relu(self.fc_post_gru(gru_out))  # (B, T, hidden_size)
        q_values = self.q_head(x)  # (B, T, action_dim)
        
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device="cpu"):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DRQLAgent(BaseAgent):
    """
    Deep Recurrent Q-Learning agent with support for flat and spatial networks.
    
    Supports two network architectures:
    - "flat": Traditional MLP (legacy, flattens spatial structure)
    - "spatial": CNN + GRU (preserves spatial structure, recommended)
    """
    def __init__(
        self, 
        player_id: int,
        obs_dim: Optional[int] = None,  # For backward compatibility with flat network
        action_dim: int = None,
        game_mode: str = "junqi_8x3",  # For spatial networks
        network_type: str = "flat",
        lr: float = 5e-4,
        gamma: float = 0.99, 
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.20,
        device: str = "cpu",
        hidden_size: int = 256,
        use_stratified_buffer: bool = True,
        num_segments: int = 4
    ):
        """
        Initialize DRQN agent.
        
        Args:
            player_id: Player ID (0 or 1)
            obs_dim: Observation dimension (only for flat network, deprecated)
            action_dim: Number of actions
            game_mode: Game mode for spatial networks ("junqi_8x3" or "junqi_standard")
            network_type: "flat" or "spatial"
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay factor per episode
            epsilon_min: Minimum epsilon value
            device: PyTorch device
            hidden_size: Hidden layer size
            use_stratified_buffer: Use temporal stratified replay buffer
            num_segments: Number of segments for stratified buffer
        """
        super().__init__(player_id, device)
        
        self.network_type = network_type
        self.game_mode = game_mode
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.hidden_size = hidden_size

        # Get board config for spatial networks
        if network_type == "spatial":
            self.board_config = get_board_config(game_mode)
            self.obs_dim = None  # Not used for spatial networks
        else:
            self.obs_dim = obs_dim
            self.board_config = None

        # Create Q-network and target network based on type
        if network_type == "flat":
            if obs_dim is None:
                raise ValueError("obs_dim required for flat network")
            self.q_net = DRQNetwork(obs_dim, action_dim, hidden_size).to(device)
            self.target_net = DRQNetwork(obs_dim, action_dim, hidden_size).to(device)
        elif network_type == "spatial":
            self.q_net = SpatialDRQNetwork(self.board_config, action_dim, hidden_size).to(device)
            self.target_net = SpatialDRQNetwork(self.board_config, action_dim, hidden_size).to(device)
        else:
            raise ValueError(f"Unknown network_type: {network_type}. Expected 'flat' or 'spatial'")
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss is more stable

        # Use temporal stratified buffer or regular buffer
        if use_stratified_buffer:
            self.replay_buffer = TemporalStratifiedReplayBuffer(maxlen=1000000, num_segments=num_segments)
        else:
            self.replay_buffer = ReplayBuffer(maxlen=100000)
        self.use_stratified_buffer = use_stratified_buffer
        
        self.hidden = None
        
        # For tracking statistics
        self.episode_rewards = []

    def reset(self):
        """Reset hidden state (implements BaseAgent.reset)."""
        self.hidden = self.q_net.init_hidden(device=self.device)
    
    # Keep old method name for backward compatibility
    def reset_hidden(self):
        """Deprecated: use reset() instead."""
        self.reset()

    def get_obs(self, state):
        """Get observation tensor from game state."""
        if self.network_type == "flat":
            # Legacy flat observation
            obs = np.array(state.observation_tensor(self.player_id), dtype=np.float32)
            return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        else:
            # Spatial observation for CNN networks
            return process_observation(
                state,
                self.player_id,
                self.board_config,
                device=self.device,
                format="spatial",
                add_batch_dim=True,
                add_seq_dim=True
            )

    def choose_action(self, state, legal_actions: Optional[List[int]] = None, eval_mode: bool = False) -> int:
        """
        Choose action (implements BaseAgent.choose_action).
        
        Args:
            state: Game state
            legal_actions: List of legal actions (will be computed if None)
            eval_mode: If True, use greedy action selection
            
        Returns:
            Action index
        """
        if legal_actions is None:
            legal_actions = state.legal_actions(self.player_id)
        
        epsilon = 0.0 if eval_mode else self.epsilon
        
        if random.random() < epsilon:
            return random.choice(legal_actions)

        obs = self.get_obs(state)
        with torch.no_grad():
            q_vals, self.hidden = self.q_net(obs, self.hidden)
            q_vals = q_vals.squeeze(0).squeeze(0)

        masked_q = torch.full_like(q_vals, -1e9)
        legal_tensor = torch.tensor(legal_actions, device=self.device)
        masked_q[legal_tensor] = q_vals[legal_tensor]
        return masked_q.argmax().item()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample based on buffer type
        if self.use_stratified_buffer:
            batch = self.replay_buffer.sample(batch_size)
        else:
            batch = random.sample(self.replay_buffer, batch_size)
        
        # Unpack batch
        obs_list = [t[0] for t in batch]
        actions = torch.tensor([t[1] for t in batch], device=self.device).long()
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=self.device)
        next_obs_list = [t[3] for t in batch]
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=self.device)

        # Stack observations (each is already (1, 1, obs_dim))
        obs_batch = torch.cat(obs_list, dim=0)  # (batch, 1, obs_dim)
        next_obs_batch = torch.cat(next_obs_list, dim=0)  # (batch, 1, obs_dim)

        # Initialize hidden states for batch processing
        batch_hidden = self.q_net.init_hidden(batch_size=batch_size, device=self.device)
        
        # Current Q values with initialized hidden state
        current_q, _ = self.q_net(obs_batch, batch_hidden)  # (batch, 1, action_dim)
        current_q = current_q.squeeze(1)  # (batch, action_dim)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Target Q values with Double DQN
        with torch.no_grad():
            # Initialize hidden for next states
            next_batch_hidden = self.q_net.init_hidden(batch_size=batch_size, device=self.device)
            
            # Use online network to select actions
            next_q_online, _ = self.q_net(next_obs_batch, next_batch_hidden)
            next_q_online = next_q_online.squeeze(1)
            next_actions = next_q_online.argmax(dim=1)
            
            # Use target network to evaluate actions
            target_batch_hidden = self.target_net.init_hidden(batch_size=batch_size, device=self.device)
            next_q_target, _ = self.target_net(next_obs_batch, target_batch_hidden)
            next_q_target = next_q_target.squeeze(1)
            next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Compute additional metrics for logging
        metrics = {
            'loss': loss.item(),
            'grad_norm_total': total_grad_norm.item(),
            'q_mean': current_q.mean().item(),
            'q_std': current_q.std().item(),
        }
        
        # Per-layer gradient norms (all layers individually)
        with torch.no_grad():
            for name, param in self.q_net.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    # Use full parameter name for detailed logging
                    clean_name = name.replace('.', '/')
                    metrics[f'grad_norm/{clean_name}'] = grad_norm
        
        # Compute action entropy from Q-values
        with torch.no_grad():
            q_vals_full, _ = self.q_net(obs_batch, batch_hidden)
            q_vals_full = q_vals_full.squeeze(1)  # (batch, action_dim)
            # Convert Q-values to probabilities using softmax
            probs = torch.softmax(q_vals_full, dim=1)
            # Compute entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            metrics['action_entropy'] = entropy.item()
            
        return metrics

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        """
        Save agent checkpoint (implements BaseAgent.save).
        
        Args:
            path: File path to save checkpoint
        """
        checkpoint = {
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'network_type': self.network_type,
            'game_mode': self.game_mode,
            'action_dim': self.action_dim,
            'hidden_size': self.hidden_size,
            'player_id': self.player_id,
            'obs_dim': self.obs_dim,  # For backward compatibility
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """
        Load agent checkpoint (implements BaseAgent.load).
        
        Args:
            path: File path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
        
        # Load network metadata for compatibility checks
        if 'network_type' in checkpoint:
            loaded_network_type = checkpoint['network_type']
            if loaded_network_type != self.network_type:
                print(f"Warning: Loading checkpoint with network_type={loaded_network_type} "
                      f"into agent with network_type={self.network_type}")