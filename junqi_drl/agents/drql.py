import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

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
                 epsilon=1.0, device="cpu", hidden_size=256):
        self.player_id = player_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995  # Slower decay
        self.epsilon_min = 0.05  # Higher minimum for exploration
        self.device = device

        self.q_net = DRQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_net = DRQNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss is more stable

        self.replay_buffer = deque(maxlen=10000)  # Larger buffer
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

    def compute_n_step_returns(self, rewards, gamma=0.99, n=5):
        """Compute n-step returns for better credit assignment"""
        n_step_rewards = []
        for i in range(len(rewards)):
            n_step_return = 0
            for j in range(min(n, len(rewards) - i)):
                n_step_return += (gamma ** j) * rewards[i + j]
            n_step_rewards.append(n_step_return)
        return n_step_rewards

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

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

        # Current Q values
        current_q, _ = self.q_net(obs_batch, None)  # (batch, 1, action_dim)
        current_q = current_q.squeeze(1)  # (batch, action_dim)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Target Q values with Double DQN
        with torch.no_grad():
            # Use online network to select actions
            next_q_online, _ = self.q_net(next_obs_batch, None)
            next_q_online = next_q_online.squeeze(1)
            next_actions = next_q_online.argmax(dim=1)
            
            # Use target network to evaluate actions
            next_q_target, _ = self.target_net(next_obs_batch, None)
            next_q_target = next_q_target.squeeze(1)
            next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q_value

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

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