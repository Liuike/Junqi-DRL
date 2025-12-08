import os
import sys
import argparse
import numpy as np
import torch
import pyspiel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from junqi_drl.agents.drql import DRQLAgent
from junqi_drl.agents.random_agent import RandomAgent
from junqi_drl.core.metrics import MetricsLogger
from junqi_drl.game import junqi_8x3
from junqi_drl.game import junqi_standard

def train_drql(
    game_mode="junqi_8x3",
    network_type="flat",
    num_episodes=5000,
    eval_every=1000,
    eval_episodes=100,
    target_update_freq=300,
    opponent_update_freq=500,
    save_dir="models",
    device="cpu",
    use_wandb=True,
    wandb_project="junqi-drql",
    wandb_run_name=None,
    lr=5e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_decay=0.9995,
    epsilon_min=0.05,
    hidden_size=256,
    batch_size=64,
    replay_buffer_size=100000,
    use_stratified_buffer=True,
    num_segments=4
):
    """
    Train DRQN agent with specified network architecture.
    
    Args:
        game_mode: "junqi_8x3" or "junqi_standard"
        network_type: "flat" or "spatial"
        num_episodes: Total training episodes
        eval_every: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        target_update_freq: Update target network every N episodes
        opponent_update_freq: Update opponent every N episodes
        save_dir: Directory to save models
        device: "cpu" or "cuda"
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_decay: Epsilon decay factor per episode
        epsilon_min: Minimum epsilon value
        hidden_size: Hidden layer size
        batch_size: Training batch size
        replay_buffer_size: Size of replay buffer
        use_stratified_buffer: Use temporal stratified replay buffer
        num_segments: Number of segments for stratified buffer
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metrics logger
    config = {
        "game_mode": game_mode,
        "network_type": network_type,
        "num_episodes": num_episodes,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
        "target_update_freq": target_update_freq,
        "opponent_update_freq": opponent_update_freq,
        "device": str(device),
        "lr": lr,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "replay_buffer_size": replay_buffer_size,
        "use_stratified_buffer": use_stratified_buffer,
        "num_segments": num_segments,
    }
    
    metrics_logger = MetricsLogger(
        use_wandb=use_wandb,
        wandb_config={
            "project": wandb_project,
            "name": wandb_run_name,
            "config": config
        }
    )
    
    game = pyspiel.load_game(game_mode)
    action_dim = game.num_distinct_actions()
    
    # Create agents based on network type
    if network_type == "flat":
        obs_dim = int(np.prod(game.observation_tensor_shape()))
        print(f"Training flat network: obs_dim={obs_dim}, action_dim={action_dim}, device={device}\n")
        
        drql_agent = DRQLAgent(
            player_id=0,
            obs_dim=obs_dim,
            action_dim=action_dim,
            network_type="flat",
            lr=lr,
            gamma=gamma,
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            device=device,
            hidden_size=hidden_size,
            use_stratified_buffer=use_stratified_buffer,
            num_segments=num_segments
        )
        
        opponent_agent = DRQLAgent(
            player_id=1,
            obs_dim=obs_dim,
            action_dim=action_dim,
            network_type="flat",
            lr=lr,
            gamma=gamma,
            epsilon=0.5,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            device=device,
            hidden_size=hidden_size,
            use_stratified_buffer=use_stratified_buffer,
            num_segments=num_segments
        )
    else:
        # Spatial network
        print(f"Training {network_type} network: game_mode={game_mode}, action_dim={action_dim}, device={device}\n")
        
        drql_agent = DRQLAgent(
            player_id=0,
            action_dim=action_dim,
            game_mode=game_mode,
            network_type=network_type,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            device=device,
            hidden_size=hidden_size,
            use_stratified_buffer=use_stratified_buffer,
            num_segments=num_segments
        )
        
        opponent_agent = DRQLAgent(
            player_id=1,
            action_dim=action_dim,
            game_mode=game_mode,
            network_type=network_type,
            lr=lr,
            gamma=gamma,
            epsilon=0.5,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            device=device,
            hidden_size=hidden_size,
            use_stratified_buffer=use_stratified_buffer,
            num_segments=num_segments
        )
    # Initialize opponent with same weights
    opponent_agent.q_net.load_state_dict(drql_agent.q_net.state_dict())
    opponent_agent.target_net.load_state_dict(drql_agent.target_net.state_dict())
    
    episode_raw_rewards = []
    episode_losses = []  # Track training losses
    episode_lengths = []  # Track episode lengths
    episode_metrics = []  # Track detailed metrics from training
    best_eval_win_rate = 0.0
    opponent_update_counter = 0  # Track opponent updates
    self_play_winrate = 0.0  # Track winrate against self-play opponent

    for episode in range(1, num_episodes + 1):
        state = game.new_initial_state()
        drql_agent.reset_hidden()
        opponent_agent.reset_hidden()
        
        # Store (obs, action) pairs for each player during the game
        transitions_p0 = []  # [(obs, action), ...]
        transitions_p1 = []
        legal_actions_p0 = []  # Track legal actions for player 0

        while not state.is_terminal():
            cur_player = state.current_player()
            if cur_player == 0:
                legal_actions = state.legal_actions(0)
                obs = drql_agent.get_obs(state)
                action = drql_agent.choose_action(state, legal_actions, eval_mode=False)
                transitions_p0.append((obs, action))
                legal_actions_p0.append(legal_actions)
                state.apply_action(action)
            else:
                legal_actions = state.legal_actions(1)
                obs = opponent_agent.get_obs(state)
                action = opponent_agent.choose_action(state, legal_actions, eval_mode=False)
                transitions_p1.append((obs, action))
                state.apply_action(action)

        raw_reward = state.returns()[0]
        game_length = len(transitions_p0)  # Number of moves by player 0
        episode_raw_rewards.append(raw_reward)
        episode_lengths.append(game_length)

        # Compute rewards for each step (sparse: only terminal gets real reward)
        # But we can assign intermediate rewards using discounting
        if raw_reward == 0:  # Draw
            final_reward_p0 = -1.0  # Strong draw penalty (as bad as losing)
            final_reward_p1 = -1.0
        else:
            final_reward_p0 = raw_reward  # +1 or -1
            final_reward_p1 = -raw_reward

        # Store ALL transitions with discounted rewards (credit assignment)
        gamma = drql_agent.gamma
        
        # Player 0 transitions
        n_p0 = len(transitions_p0)
        for i, (obs, action) in enumerate(transitions_p0):
            # Discount reward based on how far from terminal
            steps_to_end = n_p0 - i - 1
            discounted_reward = final_reward_p0 * (gamma ** steps_to_end)
            
            # Next obs is the next state player 0 saw (or dummy if terminal)
            if i + 1 < n_p0:
                next_obs = transitions_p0[i + 1][0]
                done = False
            else:
                next_obs = torch.zeros_like(obs)
                done = True
            
            drql_agent.store_transition(obs, action, discounted_reward, next_obs, done)
        
        # Player 1 transitions
        n_p1 = len(transitions_p1)
        for i, (obs, action) in enumerate(transitions_p1):
            steps_to_end = n_p1 - i - 1
            discounted_reward = final_reward_p1 * (gamma ** steps_to_end)
            
            if i + 1 < n_p1:
                next_obs = transitions_p1[i + 1][0]
                done = False
            else:
                next_obs = torch.zeros_like(obs)
                done = True
            
            opponent_agent.store_transition(obs, action, discounted_reward, next_obs, done)

        metrics = drql_agent.train(batch_size=64)
        if metrics is not None:
            episode_losses.append(metrics['loss'])
            episode_metrics.append(metrics)

        if episode % target_update_freq == 0:
            drql_agent.update_target()
            
        # Update opponent every 500 episodes to create curriculum
        if episode % 500 == 0:
            opponent_agent.q_net.load_state_dict(drql_agent.q_net.state_dict())
            opponent_agent.target_net.load_state_dict(drql_agent.target_net.state_dict())
            opponent_update_counter += 1
            print(f"  → Opponent updated (update #{opponent_update_counter})")

        if episode % 100 == 0:
            recent = episode_raw_rewards[-100:]
            wins = sum(r > 0 for r in recent)
            draws = sum(r == 0 for r in recent)
            losses = sum(r < 0 for r in recent)
            self_play_winrate = wins / len(recent)  # Winrate against self-play opponent
            avg_r = np.mean(recent)
            avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0.0
            avg_episode_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
            
            print(f"Ep {episode:4d} | Self-Play Win%: {self_play_winrate:6.2%} | "
                  f"Avg R: {avg_r:+6.3f} | W/D/L: {wins:2}/{draws:2}/{losses:2} | "
                  f"ε: {drql_agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buf: {len(drql_agent.replay_buffer)}")
            
            # Log metrics
            log_dict = {
                "episode": episode,
                "self_play_winrate": self_play_winrate,
                "self_play_wins": wins,
                "self_play_draws": draws,
                "self_play_losses": losses,
                "avg_reward": avg_r,
                "epsilon": drql_agent.epsilon,
                "avg_loss": avg_loss,
                "avg_episode_length": avg_episode_length,
                "replay_buffer_size": len(drql_agent.replay_buffer),
                "opponent_updates": opponent_update_counter
            }
            
            # Add detailed metrics from recent training steps
            if episode_metrics:
                recent_metrics = episode_metrics[-100:]
                
                # Aggregate all gradient norms from metrics
                for key in recent_metrics[0].keys():
                    if key.startswith('grad_norm'):
                        values = [m.get(key, 0) for m in recent_metrics if key in m]
                        if values:
                            log_dict[key] = np.mean(values)
                
                # Q-value statistics
                q_means = [m.get('q_mean', 0) for m in recent_metrics if 'q_mean' in m]
                q_stds = [m.get('q_std', 0) for m in recent_metrics if 'q_std' in m]
                if q_means:
                    log_dict['q_values/mean'] = np.mean(q_means)
                if q_stds:
                    log_dict['q_values/std'] = np.mean(q_stds)
                
                # Action entropy
                entropies = [m.get('action_entropy', 0) for m in recent_metrics if 'action_entropy' in m]
                if entropies:
                    log_dict['action_entropy'] = np.mean(entropies)
            
            metrics_logger.log(log_dict, step=episode)

        if episode % eval_every == 0:
            print(f"\n{'='*60}")
            print(f"EVALUATION vs RANDOM at episode {episode}")

            original_epsilon = drql_agent.epsilon
            drql_agent.epsilon = 0.0
            
            eval_wins, eval_draws, eval_losses = 0, 0, 0
            eval_opponent = RandomAgent(player_id=1)  # Evaluate against random
            for _ in range(eval_episodes):
                s = game.new_initial_state()
                drql_agent.reset_hidden()
                while not s.is_terminal():
                    p = s.current_player()
                    if p == 0:
                        a = drql_agent.choose_action(s, s.legal_actions(0), eval_mode=True)
                    else:
                        a = eval_opponent.step(s)
                    s.apply_action(a)
                r0 = s.returns()[0]
                if r0 > 0:
                    eval_wins += 1
                elif r0 == 0:
                    eval_draws += 1
                else:
                    eval_losses += 1

            eval_win_rate = eval_wins / eval_episodes
            eval_draw_rate = eval_draws / eval_episodes
            eval_loss_rate = eval_losses / eval_episodes
            
            print(f"Eval vs Random ({eval_episodes} games): "
                  f"Win: {eval_win_rate:.2%} | "
                  f"Draw: {eval_draw_rate:.2%} | "
                  f"Loss: {eval_loss_rate:.2%}")
            
            # Log evaluation winrate
            metrics_logger.log({"eval_vs_random_winrate": eval_win_rate}, step=episode)
            
            if eval_win_rate > best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                path = os.path.join(save_dir, f"drql_{network_type}_best_wr{eval_win_rate:.3f}_ep{episode}.pth")
                drql_agent.save(path)
                print(f"✓ Best model saved: {os.path.basename(path)}")
                
                # Log best model
                metrics_logger.log({"best_eval_winrate": best_eval_win_rate}, step=episode)
            
            drql_agent.epsilon = original_epsilon
            print(f"{'='*60}\n")

    drql_agent.save(os.path.join(save_dir, f"drql_{network_type}_final.pth"))
    print(f"\nTraining finished. Best eval win rate vs random: {best_eval_win_rate:.2%}")
    print(f"Final self-play winrate: {self_play_winrate:.2%}")
    
    # Finish metrics logging
    metrics_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRQN agent with configurable network architecture")
    parser.add_argument("--game_mode", type=str, default="junqi_8x3", choices=["junqi_8x3", "junqi_standard"],
                        help="Game variant to train on")
    parser.add_argument("--network_type", type=str, default="spatial", choices=["flat", "spatial"],
                        help="Network architecture: flat (MLP) or spatial (CNN)")
    parser.add_argument("--num_episodes", type=int, default=5000,
                        help="Total training episodes")
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="Evaluate every N episodes")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of episodes for evaluation")
    parser.add_argument("--target_update_freq", type=int, default=300,
                        help="Update target network every N episodes")
    parser.add_argument("--opponent_update_freq", type=int, default=500,
                        help="Update self-play opponent every N episodes")
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden layer size")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995,
                        help="Epsilon decay factor per episode")
    parser.add_argument("--epsilon_min", type=float, default=0.05,
                        help="Minimum epsilon value")
    parser.add_argument("--replay_buffer_size", type=int, default=100000,
                        help="Size of replay buffer")
    parser.add_argument("--use_stratified_buffer", action="store_true", default=True,
                        help="Use temporal stratified replay buffer")
    parser.add_argument("--num_segments", type=int, default=4,
                        help="Number of segments for stratified buffer")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="junqi-drql",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("DRQN Training Configuration")
    print("=" * 60)
    print(f"Game Mode: {args.game_mode}")
    print(f"Network Type: {args.network_type}")
    print(f"Device: {device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"WandB Enabled: {args.wandb}")
    print("=" * 60)
    print()
    
    train_drql(
        game_mode=args.game_mode,
        network_type=args.network_type,
        num_episodes=args.num_episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        target_update_freq=args.target_update_freq,
        opponent_update_freq=args.opponent_update_freq,
        save_dir=args.save_dir,
        device=device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        use_stratified_buffer=args.use_stratified_buffer,
        num_segments=args.num_segments
    )
