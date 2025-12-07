import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class TemporalStratifiedReplayBuffer:
    """
    Temporal stratified replay buffer that divides the buffer into time segments
    and samples proportionally from each segment to ensure early experiences are represented.
    """
    def __init__(self, maxlen=100000, num_segments=4):
        """
        Args:
            maxlen: Maximum buffer size
            num_segments: Number of temporal segments to divide the buffer into
        """
        self.maxlen = maxlen
        self.num_segments = num_segments
        self.segment_size = maxlen // num_segments
        
        # Create separate deques for each temporal segment
        self.segments = [deque(maxlen=self.segment_size) for _ in range(num_segments)]
        self.current_segment = 0
        self.total_added = 0
        
    def append(self, transition):
        """
        Add transition to the current segment.
        Advances to next segment when current is full.
        """
        self.segments[self.current_segment].append(transition)
        self.total_added += 1
        
        # Move to next segment in round-robin fashion when segment is full
        if len(self.segments[self.current_segment]) >= self.segment_size:
            self.current_segment = (self.current_segment + 1) % self.num_segments
    
    def sample(self, batch_size):
        """
        Sample uniformly from all temporal segments.
        Each segment contributes proportionally to its size.
        """
        batch = []
        
        # Count non-empty segments
        non_empty_segments = [seg for seg in self.segments if len(seg) > 0]
        
        if not non_empty_segments:
            return batch
        
        # Calculate samples per segment (uniform across non-empty segments)
        samples_per_segment = batch_size // len(non_empty_segments)
        remainder = batch_size % len(non_empty_segments)
        
        for i, segment in enumerate(non_empty_segments):
            # Add extra sample to first 'remainder' segments to reach exact batch_size
            n_samples = samples_per_segment + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(segment))
            
            batch.extend(random.sample(list(segment), n_samples))
        
        # If we still don't have enough, sample with replacement from all segments
        if len(batch) < batch_size:
            all_transitions = []
            for segment in non_empty_segments:
                all_transitions.extend(list(segment))
            
            if all_transitions:
                remaining = batch_size - len(batch)
                batch.extend(random.choices(all_transitions, k=remaining))
        
        return batch
    
    def __len__(self):
        """Total number of transitions across all segments."""
        return sum(len(seg) for seg in self.segments)
    
    def get_stats(self):
        """Return statistics about buffer composition."""
        stats = {
            'total': len(self),
            'segments': []
        }
        
        for i, segment in enumerate(self.segments):
            stats['segments'].append({
                'id': i,
                'size': len(segment),
                'is_current': i == self.current_segment
            })
        
        return stats

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


class DRQLAgent:
    def __init__(self, player_id, obs_dim, action_dim, lr=5e-4, gamma=0.99, 
                 epsilon=1.0, device="cpu", hidden_size=256, use_stratified_buffer=True, num_segments=4):
        self.player_id = player_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995 
        self.epsilon_min = 0.20
        self.device = device

        self.q_net = DRQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_net = DRQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss is more stable

        # Use temporal stratified buffer or regular deque
        if use_stratified_buffer:
            self.replay_buffer = TemporalStratifiedReplayBuffer(maxlen=1000000, num_segments=num_segments)
        else:
            self.replay_buffer = deque(maxlen=100000)
        self.use_stratified_buffer = use_stratified_buffer
        
        self.hidden = None
        
        # For tracking statistics
        self.episode_rewards = []

    def reset_hidden(self):
        self.hidden = self.q_net.init_hidden(device=self.device)

    def get_obs(self, state):
        obs = np.array(state.observation_tensor(self.player_id), dtype=np.float32)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    def choose_action(self, state, legal_actions, eval_mode=False):
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

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']