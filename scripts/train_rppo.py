import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import pyspiel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, metrics will not be logged to W&B")

from junqi_drl.agents.rppo import RPPoAgent
from junqi_drl.agents.random_agent import RandomAgent
from junqi_drl.game import junqi_8x3  # noqa: F401
from junqi_drl.game import junqi_standard  # noqa: F401

# Default to the small 8x3 board for training
gamemode = "junqi_8x3"


def evaluate_agent(
    agent: RPPoAgent,
    game,
    num_episodes: int = 50,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """
    Evaluate the given agent against a random opponent.

    Returns a dict with win/draw/loss rates and avg game length.
    """
    agent_device = torch.device(device)
    agent.net.to(agent_device)

    random_agent = RandomAgent(player_id=1)

    wins = draws = losses = 0
    total_moves = 0

    for _ in range(num_episodes):
        state = game.new_initial_state()
        agent.reset_hidden()
        random_agent.reset()
        move_count = 0

        while not state.is_terminal():
            cur_player = state.current_player()

            if cur_player == 0:
                legal_actions = state.legal_actions(0)
                if not legal_actions:
                    break

                action = agent.choose_action(
                    state, legal_actions, eval_mode=True
                )
                state.apply_action(action)
            else:
                action = random_agent.step(state)
                if action is None:
                    break
                state.apply_action(action)

            move_count += 1

        returns = state.returns()
        reward_p0 = returns[0]

        if reward_p0 > 0:
            wins += 1
        elif reward_p0 < 0:
            losses += 1
        else:
            draws += 1

        total_moves += move_count

    win_rate = wins / num_episodes
    draw_rate = draws / num_episodes
    loss_rate = losses / num_episodes
    avg_moves = total_moves / num_episodes

    return {
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "avg_moves": avg_moves,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


def train_rppo(
    num_episodes: int = 5000,
    eval_every: int = 1000,
    eval_episodes: int = 100,
    save_dir: str = "models",
    device: str | torch.device = "cpu",
    use_wandb: bool = True,
    wandb_project: str = "junqi-rppo",
    wandb_run_name: str | None = None,
    # RPPO hyperparameters (MUST match train_from_config.py!)
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    k_epochs: int = 4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    hidden_size: int = 256,
    batch_size: int = 64,
):
    os.makedirs(save_dir, exist_ok=True)

    # Initialize wandb (similar spirit to transformer/drql)
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "algo": "R-PPO",
                "game_mode": gamemode,
                "num_episodes": num_episodes,
                "eval_every": eval_every,
                "eval_episodes": eval_episodes,
                "lr": lr,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_eps": clip_eps,
                "k_epochs": k_epochs,
                "value_coef": value_coef,
                "entropy_coef": entropy_coef,
                "max_grad_norm": max_grad_norm,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "device": str(device),
            },
        )
        use_wandb = True
    else:
        use_wandb = False

    # Create game from OpenSpiel registration
    game = pyspiel.load_game(gamemode)
    obs_dim = int(np.prod(game.observation_tensor_shape()))
    action_dim = game.num_distinct_actions()

    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, Device: {device}\n")

    device = torch.device(device)

    # Main R-PPO agent plays as player 0
    rppo_agent = RPPoAgent(
        player_id=0,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        k_epochs=k_epochs,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        device=device,
        hidden_size=hidden_size,
    )

    # Opponent: simple random agent on player 1
    opponent_agent = RandomAgent(player_id=1)

    episode_raw_rewards: List[float] = []
    episode_lengths: List[int] = []

    best_eval_win_rate = 0.0

    for episode in range(1, num_episodes + 1):
        state = game.new_initial_state()
        rppo_agent.reset_hidden()
        opponent_agent.reset()

        # Store (obs, action) for player 0 transitions
        transitions_p0: List[Tuple[torch.Tensor, int]] = []

        while not state.is_terminal():
            cur_player = state.current_player()

            if cur_player == 0:
                legal_actions = state.legal_actions(0)
                if not legal_actions:
                    break

                obs = rppo_agent.get_obs(state)
                action = rppo_agent.choose_action(
                    state, legal_actions, eval_mode=False
                )

                transitions_p0.append((obs, action))
                state.apply_action(action)
            else:
                action = opponent_agent.step(state)
                if action is None:
                    break
                state.apply_action(action)

        # Terminal reward from the game
        raw_reward = state.returns()[0]
        game_length = len(transitions_p0)

        episode_raw_rewards.append(raw_reward)
        episode_lengths.append(game_length)

        if raw_reward == 0:
            final_reward_p0 = -1.0
        else:
            final_reward_p0 = raw_reward  # +1 or -1

        gamma_discount = rppo_agent.gamma
        n_p0 = len(transitions_p0)

        for i, (obs, action) in enumerate(transitions_p0):
            steps_to_end = n_p0 - i - 1
            discounted_reward = final_reward_p0 * (gamma_discount ** steps_to_end)

            if i + 1 < n_p0:
                next_obs = transitions_p0[i + 1][0]
                done = False
            else:
                next_obs = torch.zeros_like(obs)
                done = True

            rppo_agent.store_transition(
                obs, action, discounted_reward, next_obs, done
            )

        metrics = rppo_agent.train(batch_size=batch_size)

        if metrics is None:
            print(f"Episode {episode}: no update (not enough data)")
        else:
            avg_reward = metrics.get("avg_reward", 0.0)
            print(
                f"Episode {episode}: raw_reward={raw_reward:.2f}, "
                f"avg_reward_batch={avg_reward:.3f}, "
                f"steps={metrics.get('num_steps', 0)}"
            )

        # Periodic evaluation vs random opponent
        if eval_every > 0 and episode % eval_every == 0:
            print(f"\n=== Evaluation after episode {episode} ===")
            eval_stats = evaluate_agent(
                rppo_agent,
                game,
                num_episodes=eval_episodes,
                device=device,
            )

            win_rate = eval_stats["win_rate"]

            print(
                f"Eval vs random: win_rate={win_rate:.3f}, "
                f"draw_rate={eval_stats['draw_rate']:.3f}, "
                f"loss_rate={eval_stats['loss_rate']:.3f}, "
                f"avg_moves={eval_stats['avg_moves']:.1f}"
            )

            # Save best-performing model
            if win_rate > best_eval_win_rate:
                best_eval_win_rate = win_rate
                best_path = os.path.join(save_dir, "rppo_best.pt")
                rppo_agent.save(best_path)
                print(
                    f" → New best model saved to {best_path} "
                    f"(win_rate={win_rate:.3f})"
                )

            # Log eval metrics to wandb (step = episode like transformer/drql)
            if use_wandb:
                wandb.log(
                    {
                        "episode": episode,
                        "eval_vs_random/winrate": win_rate,
                        "eval_vs_random/drawrate": eval_stats["draw_rate"],
                        "eval_vs_random/lossrate": eval_stats["loss_rate"],
                        "eval_vs_random/wins": eval_stats["wins"],
                        "eval_vs_random/draws": eval_stats["draws"],
                        "eval_vs_random/losses": eval_stats["losses"],
                        "eval_vs_random/avg_moves": eval_stats["avg_moves"],
                    },
                    step=episode,
                )

        # Log training metrics to wandb every episode
        if use_wandb and metrics is not None:
            log_dict: Dict[str, float] = {
                "episode": episode,
                "raw_reward": raw_reward,
                "episode_length": game_length,
                "best_eval_win_rate": best_eval_win_rate,
            }

            # Merge PPO-specific metrics (losses, entropy, etc.)
            for k, v in metrics.items():
                log_dict[f"train/{k}"] = v

            # This is what creates the training curves in wandb
            wandb.log(log_dict, step=episode)

        # Periodically save checkpoint
        if episode % 500 == 0:
            ckpt_path = os.path.join(save_dir, f"rppo_ep{episode}.pt")
            rppo_agent.save(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()
