import os
import sys
import numpy as np
import torch
import pyspiel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from junqi_drl.agents.drql import DRQLAgent
from junqi_drl.agents.random_agent import RandomAgent
from junqi_drl.game import junqi_8x3
from junqi_drl.game import junqi_standard

gamemode = "junqi_standard"

def train_drql(
    num_episodes=3000,
    eval_every=1000,
    eval_episodes=100,
    target_update_freq=1000,
    save_dir="models",
    device="cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    
    game = pyspiel.load_game(gamemode)
    obs_dim = int(np.prod(game.observation_tensor_shape()))
    action_dim = game.num_distinct_actions()
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, Device: {device}\n")
    
    drql_agent = DRQLAgent(
        player_id=0,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        device=device,
        hidden_size=256
    )

    #AGENT
    agent = RandomAgent(player_id=1)
    
    episode_raw_rewards = []
    best_eval_win_rate = 0.0

    for episode in range(1, num_episodes + 1):
        state = game.new_initial_state()
        drql_agent.reset_hidden()
        obs_history = []

        while not state.is_terminal():
            cur_player = state.current_player()
            if cur_player == 0:
                legal_actions = state.legal_actions(0)
                action = drql_agent.choose_action(state, legal_actions, eval_mode=False)
                obs_history.append(drql_agent.get_obs(state))
            else:
                action = agent.step(state)
            state.apply_action(action)

        raw_reward = state.returns()[0]
        game_length = state.game_length
        episode_raw_rewards.append(raw_reward)

        if raw_reward == 0:  # Draw
            shaped_reward = -0.05  # Penalize draws
        else:
            # Encourage faster decisive outcomes
            time_penalty = 0.0005 * game_length
            shaped_reward = raw_reward - time_penalty

        if obs_history:
            last_obs = obs_history[-1]
            dummy_obs = torch.zeros_like(last_obs)
            drql_agent.store_transition(last_obs, action, shaped_reward, dummy_obs, True)

        drql_agent.train(batch_size=64)

        if episode % target_update_freq == 0:
            drql_agent.update_target()

        if episode % 100 == 0:
            recent = episode_raw_rewards[-100:]
            wins = sum(r > 0 for r in recent)
            draws = sum(r == 0 for r in recent)
            losses = sum(r < 0 for r in recent)
            win_rate = wins / len(recent)
            avg_r = np.mean(recent)
            print(f"Ep {episode:4d} | Win%: {win_rate:6.2%} | "
                  f"Avg R: {avg_r:+6.3f} | W/D/L: {wins:2}/{draws:2}/{losses:2} | "
                  f"ε: {drql_agent.epsilon:.3f} | "
                  f"Buf: {len(drql_agent.replay_buffer)}")

        if episode % eval_every == 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION at episode {episode}")

            original_epsilon = drql_agent.epsilon
            drql_agent.epsilon = 0.0
            
            eval_wins, eval_draws, eval_losses = 0, 0, 0
            for _ in range(eval_episodes):
                s = game.new_initial_state()
                drql_agent.reset_hidden()
                while not s.is_terminal():
                    p = s.current_player()
                    if p == 0:
                        a = drql_agent.choose_action(s, s.legal_actions(0), eval_mode=True)
                    else:
                        a = agent.step(s)
                    s.apply_action(a)
                r0 = s.returns()[0]
                if r0 > 0:
                    eval_wins += 1
                elif r0 == 0:
                    eval_draws += 1
                else:
                    eval_losses += 1

            eval_win_rate = eval_wins / eval_episodes
            print(f"Eval Results ({eval_episodes} games): "
                  f"Win: {eval_win_rate:.2%} | "
                  f"Draw: {eval_draws/eval_episodes:.2%} | "
                  f"Loss: {eval_losses/eval_episodes:.2%}")
            
            if eval_win_rate > best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                path = os.path.join(save_dir, f"drql_best_wr{eval_win_rate:.3f}_ep{episode}.pth")
                drql_agent.save(path)
                print(f"model saved: {os.path.basename(path)}")
            
            drql_agent.epsilon = original_epsilon
            print(f"{'='*60}\n")

    drql_agent.save(os.path.join(save_dir, "drql_final.pth"))
    print(f"\nTraining finished. Best eval win rate: {best_eval_win_rate:.2%}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_drql(
        num_episodes=3000,
        eval_every=1000,
        eval_episodes=100,
        target_update_freq=1000,
        save_dir="models",
        device=device
    )