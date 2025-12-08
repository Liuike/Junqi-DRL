"""
Simple reinforcement learning training script for Junqi (junqi_standard).

This uses a very basic linear Q-learning agent for Player 0,
training against a random opponent (Player 1).

Requirements (install via pip if needed):

    pip install open-spiel-junqi

Run from the project root (Junqi-DRL directory), e.g.:

    python examples/train_simple_q_learning.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add the project root (Junqi-DRL) to the path so we can import junqi_drl.*
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel  # provided by open-spiel-junqi
from junqi_drl.game import junqi_standard  # noqa: F401  # registers the game
from junqi_drl.agents.random_agent import RandomAgent


class LinearQAgent:
    """
    Very simple linear function-approximation Q-learning agent.

    Q(s, a) = w_a^T * phi(s)

    where phi(s) is the flattened observation tensor for the current player.
    """

    def __init__(
        self,
        player_id: int,
        obs_dim: int,
        num_actions: int,
        alpha: float = 0.01,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_episodes: int = 500,
        seed: int | None = None,
    ) -> None:
        self.player_id = player_id
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = max(1, epsilon_decay_episodes)
        self.total_episodes = 0

        # Weights: shape (num_actions, obs_dim)
        rng = np.random.default_rng(seed)
        # small Gaussian init
        self.weights = rng.normal(
            loc=0.0, scale=0.01, size=(num_actions, obs_dim)
        ).astype(np.float32)

    def current_epsilon(self) -> float:
        """Linearly decaying epsilon schedule."""
        frac = min(1.0, self.total_episodes / self.epsilon_decay_episodes)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute Q(s, ·) for a given observation.

        Args:
            obs: 1D numpy array of shape (obs_dim,)

        Returns:
            q: 1D numpy array of shape (num_actions,)
        """
        return self.weights @ obs  # (num_actions, obs_dim) @ (obs_dim,) -> (num_actions,)

    def select_action(self, obs: np.ndarray, legal_actions: List[int], epsilon: float) -> int:
        """
        Epsilon-greedy action selection restricted to legal actions.
        """
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        if np.random.rand() < epsilon:
            # Uniform random among legal actions
            return int(np.random.choice(legal_actions))

        q_vals = self.q_values(obs)
        # Choose the legal action with maximum Q-value
        best_action = max(legal_actions, key=lambda a: q_vals[a])
        return int(best_action)

    def update_episode(
        self,
        trajectory: List[Tuple[np.ndarray, int]],
        final_return: float,
    ) -> None:
        """
        Monte Carlo update for all (s_t, a_t) visited by this agent in one episode.

        Args:
            trajectory: list of (obs, action) pairs for this agent (only Player 0 here)
            final_return: scalar return for this agent at episode end (e.g., +1, -1, or 0)
        """
        for obs, action in trajectory:
            # Current estimate Q(s, a)
            q_sa = float(self.weights[action].dot(obs))
            # Monte Carlo target is the final return (no intermediate rewards)
            td_error = final_return - q_sa
            # Gradient of Q wrt weights[action] is obs
            self.weights[action] += self.alpha * td_error * obs


def run_training(
    num_episodes: int = 1000,
    alpha: float = 0.01,
    gamma: float = 0.99,
    seed: int | None = 42,
) -> None:
    """
    Train a LinearQAgent (Player 0) against a RandomAgent (Player 1) on junqi_standard.
    """

    np.random.seed(seed)

    # Ensure game is registered by importing junqi_standard (import above)
    game = pyspiel.load_game("junqi_standard")

    num_actions = game.num_distinct_actions()
    obs_shape = game.observation_tensor_shape()
    obs_dim = int(np.prod(obs_shape))

    print(f"Loaded game: {game.get_type().long_name}")
    print(f"Observation shape: {obs_shape}, flattened dim = {obs_dim}")
    print(f"Number of distinct actions: {num_actions}")
    print()

    # Create agents
    q_agent = LinearQAgent(
        player_id=0,
        obs_dim=obs_dim,
        num_actions=num_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_episodes=max(1, num_episodes // 2),
        seed=seed,
    )
    random_agent_1 = RandomAgent(player_id=1)

    # Training loop
    win_count = 0
    draw_count = 0
    loss_count = 0

    for episode in range(1, num_episodes + 1):
        state = game.new_initial_state()
        q_agent_trajectory: List[Tuple[np.ndarray, int]] = []

        # Episode rollout
        while not state.is_terminal():
            current_player = state.current_player()

            # Our learning agent (Player 0)
            if current_player == q_agent.player_id:
                obs = np.array(
                    state.observation_tensor(q_agent.player_id),
                    dtype=np.float32,
                ).reshape(-1)

                legal_actions = state.legal_actions()
                epsilon = q_agent.current_epsilon()
                action = q_agent.select_action(obs, legal_actions, epsilon)

                q_agent_trajectory.append((obs, action))
                state.apply_action(action)

            # Opponent: random agent (Player 1)
            elif current_player == 1:
                action = random_agent_1.step(state)
                # In case no legal action (should not happen normally), just break
                if action is None:
                    break
                state.apply_action(action)
            else:
                # Should not happen for a two-player, deterministic game
                raise RuntimeError(f"Unexpected current_player: {current_player}")

        # Episode ended
        returns = state.returns()
        final_return_p0 = returns[0]

        if final_return_p0 > 0:
            win_count += 1
        elif final_return_p0 < 0:
            loss_count += 1
        else:
            draw_count += 1

        # Monte Carlo update for Player 0
        q_agent.update_episode(q_agent_trajectory, final_return_p0)
        q_agent.total_episodes += 1

        # Logging
        if episode % max(1, num_episodes // 10) == 0 or episode == 1:
            eps_val = q_agent.current_epsilon()
            total = win_count + draw_count + loss_count
            win_rate = win_count / total if total > 0 else 0.0
            print(
                f"Episode {episode:5d}/{num_episodes} | "
                f"eps={eps_val:.3f} | "
                f"W/D/L = {win_count}/{draw_count}/{loss_count} "
                f"(win_rate={win_rate:.3f})"
            )

    print("\nTraining complete.")
    total = win_count + draw_count + loss_count
    if total > 0:
        print(
            f"Final results over {total} episodes vs random: "
            f"W={win_count}, D={draw_count}, L={loss_count}, "
            f"win_rate={win_count / total:.3f}"
        )
    else:
        print("No episodes played? Something went wrong.")


def evaluate_agent(
    q_agent: LinearQAgent,
    game: pyspiel.Game,
    num_games: int = 20,
    seed: int | None = 123,
) -> None:
    """
    Optional extra evaluation function if you want to call it from a Python shell.

    Plays the trained LinearQAgent (Player 0) against a fresh RandomAgent (Player 1)
    using greedy (epsilon=0) actions.
    """
    np.random.seed(seed)
    random_agent_1 = RandomAgent(player_id=1)

    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        state = game.new_initial_state()

        while not state.is_terminal():
            current_player = state.current_player()
            if current_player == q_agent.player_id:
                obs = np.array(
                    state.observation_tensor(q_agent.player_id),
                    dtype=np.float32,
                ).reshape(-1)
                legal_actions = state.legal_actions()
                # Greedy: epsilon = 0
                action = q_agent.select_action(obs, legal_actions, epsilon=0.0)
                state.apply_action(action)
            else:
                action = random_agent_1.step(state)
                if action is None:
                    break
                state.apply_action(action)

        ret = state.returns()[0]
        if ret > 0:
            wins += 1
        elif ret < 0:
            losses += 1
        else:
            draws += 1

    print(
        f"Evaluation over {num_games} games vs random: "
        f"W={wins}, D={draws}, L={losses}, win_rate={wins / num_games:.3f}"
    )


if __name__ == "__main__":
    # You can adjust hyperparameters here as desired.
    run_training(
        num_episodes=500,  # try increasing to 5000+ for more learning
        alpha=0.01,
        gamma=0.99,
        seed=42,
    )
