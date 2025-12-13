import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Optional


class RPPONetwork(nn.Module):
    """
    Recurrent policy + value network used by the R-PPO agent.

    The network takes an observation sequence of shape (batch, T, obs_dim),
    runs it through a GRU, and produces:
      - action logits of shape (batch, action_dim)
      - state-value estimates of shape (batch,)
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Initialize a zero hidden state for the GRU.

        Returns:
            hidden: (num_layers=1, batch_size, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """
        Args:
            obs: Tensor of shape (batch, T, obs_dim)
            hidden: initial hidden state (1, batch, hidden_size)

        Returns:
            logits: (batch, action_dim)
            values: (batch,)
            next_hidden: (1, batch, hidden_size)
        """
        batch_size, T, _ = obs.shape
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        out, next_hidden = self.gru(x, hidden)

        last = out[:, -1, :]
        logits = self.policy_head(last)
        values = self.value_head(last).squeeze(-1)

        return logits, values, next_hidden


class RPPORolloutBuffer:
    """
    Simple on-policy rollout buffer for PPO.

    We store transitions as flat vectors (no batching here). For this project
    we assume that each call to `store_transition` corresponds to a single
    environment step, and that episodes are delineated using the `done` flag.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, logprob, reward, done, value):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.obs)


class RPPoAgent:
    """
    Recurrent Proximal Policy Optimization (R-PPO) agent.

    This agent is designed to fit cleanly into the existing project structure:
      - It uses the same observation encoding as DRQLAgent (state.observation_tensor).
      - It exposes a similar interface: reset_hidden, get_obs, choose_action,
        store_transition, and train.

    Typical usage pattern:

        agent.reset_hidden()
        while not state.is_terminal():
            legal_actions = state.legal_actions(cur_player)
            action = agent.choose_action(state, legal_actions)
            next_state = ...
            agent.store_transition(obs, action, reward, next_obs, done)

        metrics = agent.train()
    """
    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        k_epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
        hidden_size: int = 256,
    ):
        self.player_id = player_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device)

        self.net = RPPONetwork(obs_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.buffer = RPPORolloutBuffer()

        self.hidden = self.net.init_hidden(batch_size=1, device=self.device)

    def reset_hidden(self):
        """Reset recurrent hidden state at the start of an episode."""
        self.hidden = self.net.init_hidden(batch_size=1, device=self.device)

    def get_obs(self, state) -> torch.Tensor:
        """
        Encode state into a torch tensor, matching DRQLAgent convention.

        Returns:
            obs: shape (1, 1, obs_dim) on the correct device.
        """
        obs = np.array(state.observation_tensor(self.player_id), dtype=np.float32)
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def _policy_value(self, obs_tensor: torch.Tensor, hidden: torch.Tensor):
        """
        Helper to run a single-step forward pass through the recurrent net.

        Args:
            obs_tensor: (1, 1, obs_dim)
            hidden: (1, 1, hidden_size)

        Returns:
            logits: (action_dim,)
            value: scalar tensor
            next_hidden: (1, 1, hidden_size)
        """
        logits, values, next_hidden = self.net(obs_tensor, hidden)
        # Remove batch dimension
        logits = logits.squeeze(0)   # (action_dim,)
        value = values.squeeze(0)    # ()
        return logits, value, next_hidden

    def choose_action(self, state, legal_actions: List[int], eval_mode: bool = False) -> int:
        """
        Select an action using the current policy.

        Args:
            state: pyspiel.State
            legal_actions: list of legal action indices
            eval_mode: if True, use greedy (argmax) instead of sampling

        Returns:
            action (int)
        """
        obs = self.get_obs(state)  # (1, 1, obs_dim)
        with torch.no_grad():
            logits, value, next_hidden = self._policy_value(obs, self.hidden)
        self.hidden = next_hidden

        # Mask illegal actions
        logits_masked = torch.full_like(logits, -1e9)
        if len(legal_actions) > 0:
            legal_idx = torch.tensor(legal_actions, device=self.device, dtype=torch.long)
            logits_masked[legal_idx] = logits[legal_idx]

        probs = torch.softmax(logits_masked, dim=-1)

        if eval_mode:
            action = torch.argmax(probs).item()
            logprob = torch.log(probs[action] + 1e-10).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()
            logprob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        # Cache quantities so store_transition can add them to the buffer.
        self._last_obs = obs.detach().cpu().numpy().reshape(-1)  # (obs_dim,)
        self._last_action = action
        self._last_logprob = logprob
        self._last_value = value.detach().cpu().item()

        return action

    def store_transition(self, obs, action, reward: float, next_obs, done: bool):
        """
        Store transition in the on-policy buffer.

        Note:
            For compatibility with the DRQL training loop signature, we accept
            `obs`, `action`, `reward`, `next_obs`, and `done`, but internally we
            use the quantities cached in `choose_action` (which correspond to
            the actual policy/value for that state).

        Args:
            obs: torch.Tensor or np.ndarray for current observation
            action: int
            reward: scalar reward
            next_obs: (unused for PPO but kept for interface compatibility)
            done: boolean flag
        """
        # Prefer the cached values from choose_action; fall back to `obs`/`action`
        if hasattr(self, "_last_obs"):
            obs_vec = self._last_obs
            act = self._last_action
            logprob = self._last_logprob
            value = self._last_value
        else:
            # Fallback: derive obs and values directly (slower, but robust)
            if isinstance(obs, torch.Tensor):
                obs_vec = obs.detach().cpu().numpy().reshape(-1)
            else:
                obs_vec = np.array(obs, dtype=np.float32).reshape(-1)

            obs_tensor = torch.tensor(
                obs_vec, dtype=torch.float32, device=self.device
            ).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits, value_tensor, _ = self._policy_value(
                    obs_tensor, self.net.init_hidden(batch_size=1, device=self.device)
                )
                probs = torch.softmax(logits, dim=-1)
                act = int(action)
                logprob = torch.log(probs[act] + 1e-10).item()
                value = value_tensor.detach().cpu().item()

        self.buffer.add(obs_vec, act, logprob, reward, done, value)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE-Lambda).

        Args:
            rewards: (T,) tensor
            values: (T,) tensor
            dones: (T,) tensor of {0,1}

        Returns:
            returns: (T,) tensor with discounted returns
            advantages: (T,) tensor with advantage estimates (normalized)
        """
        T = rewards.size(0)
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def train(self, batch_size: Optional[int] = None):
        """
        Perform one PPO update using the data currently in the rollout buffer.

        Args:
            batch_size: mini-batch size. If None, use all data in one batch.

        Returns:
            metrics dict (losses, stats) or None if not enough data.
        """
        if len(self.buffer) == 0:
            return None

        obs = torch.tensor(np.stack(self.buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)

        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)

        N = obs.size(0)
        if batch_size is None or batch_size > N:
            batch_size = N

        # For RNN simplicity, we treat each step as a single-element "sequence"
        # when training. This still lets the GRU be used, but without complex
        # sequence batching logic.
        def make_batch(idx):
            # obs[idx]: (obs_dim,) -> (batch, T=1, obs_dim)
            o = obs[idx].unsqueeze(1)  # (batch, 1, obs_dim)
            h0 = self.net.init_hidden(batch_size=o.size(0), device=self.device)
            return o, h0

        indices = np.arange(N)
        metrics: dict[str, float] = {}

        for epoch in range(self.k_epochs):
            np.random.shuffle(indices)

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            num_batches = 0

            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                b_actions = actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                batch_obs, batch_hidden = make_batch(batch_idx)

                logits, values_pred, _ = self.net(batch_obs, batch_hidden)
                # logits: (batch, action_dim), values_pred: (batch,)

                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO objective
                ratios = torch.exp(new_logprobs - b_old_logprobs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = (b_returns - values_pred).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                num_batches += 1

            # Average over batches for this epoch
            epoch_key = f"epoch_{epoch}"
            metrics[f"{epoch_key}/policy_loss"] = epoch_policy_loss / max(num_batches, 1)
            metrics[f"{epoch_key}/value_loss"] = epoch_value_loss / max(num_batches, 1)
            metrics[f"{epoch_key}/entropy"] = epoch_entropy / max(num_batches, 1)

        # Some global summary metrics
        metrics["num_steps"] = int(N)
        metrics["avg_return"] = returns.mean().item()
        metrics["avg_reward"] = rewards.mean().item()
        metrics["avg_value"] = values.mean().item()

        # Clear buffer after update (on-policy)
        self.buffer.clear()

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint helpers (optional)
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save network and optimizer state."""
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        """Load network and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["net"], strict=strict)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
