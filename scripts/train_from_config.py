import os
import sys
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from junqi_drl.core.config import Config
from scripts.train_dqrl import train_drql
from scripts.train_transformer import train_transformer
from scripts.train_rppo import train_rppo 


def main():
    parser = argparse.ArgumentParser(
        description="Train DRQN or Transformer agent from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--override", type=str, nargs="*", metavar="KEY=VALUE",
                        help="Override config values (e.g., training.num_iterations=1000)")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Apply overrides if provided
    if args.override:
        print("\nApplying overrides:")
        for override in args.override:
            if '=' not in override:
                print(f"  Warning: Invalid override format '{override}', skipping")
                continue
            
            key, value = override.split('=', 1)
            
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, AttributeError):
                pass  # Keep as string
            
            # Apply override to nested config
            parts = key.split('.')
            if len(parts) == 2:
                section, param = parts
                if section == 'game' and hasattr(config.game, param):
                    setattr(config.game, param, value)
                    print(f"  {key} = {value}")
                elif section == 'training' and hasattr(config.training, param):
                    setattr(config.training, param, value)
                    print(f"  {key} = {value}")
                elif section == 'wandb' and hasattr(config.wandb, param):
                    setattr(config.wandb, param, value)
                    print(f"  {key} = {value}")
                elif section == 'agent' and hasattr(config.agent_config, param):
                    setattr(config.agent_config, param, value)
                    print(f"  {key} = {value}")
                else:
                    print(f"  Warning: Unknown config key '{key}', skipping")
            else:
                print(f"  Warning: Override must be in format 'section.key=value', got '{override}'")
    
    # Auto-detect device
    device_str = config.training.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    if config.agent_type == "drqn":
        agent_cfg = config.agent_config
        
        # Print configuration
        print("\n" + "=" * 60)
        print("DRQN Training Configuration")
        print("=" * 60)
        print(f"Game Mode: {config.game.mode}")
        print(f"Network Type: {agent_cfg.network_type}")
        print(f"Device: {device}")
        print(f"Episodes: {config.training.num_episodes}")
        print(f"Hidden Size: {agent_cfg.hidden_size}")
        print(f"Learning Rate: {agent_cfg.lr}")
        print(f"Gamma: {agent_cfg.gamma}")
        print(f"Batch Size: {agent_cfg.batch_size}")
        print(f"Target Update Freq: {agent_cfg.target_update_freq}")
        print(f"Opponent Update Freq: {agent_cfg.opponent_update_freq}")
        print(f"WandB Enabled: {config.wandb.enabled}")
        if config.wandb.enabled:
            print(f"WandB Project: {config.wandb.project}")
            if config.wandb.run_name:
                print(f"WandB Run: {config.wandb.run_name}")
        print("=" * 60)
        print()
        
        # Start DRQN training
        train_drql(
            game_mode=config.game.mode,
            network_type=agent_cfg.network_type,
            num_episodes=config.training.num_episodes,
            eval_every=config.training.eval_every,
            eval_episodes=config.training.eval_episodes,
            target_update_freq=agent_cfg.target_update_freq,
            opponent_update_freq=agent_cfg.opponent_update_freq,
            batch_size=agent_cfg.batch_size,
            save_dir=config.training.save_dir,
            device=device,
            lr=agent_cfg.lr,
            gamma=agent_cfg.gamma,
            epsilon_start=agent_cfg.epsilon_start,
            epsilon_decay=agent_cfg.epsilon_decay,
            epsilon_min=agent_cfg.epsilon_min,
            hidden_size=agent_cfg.hidden_size,
            replay_buffer_size=agent_cfg.replay_buffer_size,
            use_stratified_buffer=agent_cfg.use_stratified_buffer,
            num_segments=agent_cfg.num_segments,
            use_wandb=config.wandb.enabled,
            wandb_project=config.wandb.project,
            wandb_run_name=config.wandb.run_name
        )
        
    elif config.agent_type == "transformer":
        agent_cfg = config.agent_config
        
        # Print configuration
        print("\n" + "=" * 60)
        print("Transformer PPO Training Configuration")
        print("=" * 60)
        print(f"Game Mode: {config.game.mode}")
        print(f"Device: {device}")
        print(f"Iterations: {config.training.num_iterations}")
        print(f"Steps per Iteration: {agent_cfg.num_steps}")
        print(f"Model Dimension: {agent_cfg.d_model}")
        print(f"Attention Heads: {agent_cfg.nhead}")
        print(f"Transformer Layers: {agent_cfg.num_layers}")
        print(f"Learning Rate: {agent_cfg.lr_start} → {agent_cfg.lr_end}")
        print(f"Entropy Coef: {agent_cfg.ent_coef_start} → {agent_cfg.ent_coef_end}")
        print(f"WandB Enabled: {config.wandb.enabled}")
        if config.wandb.enabled:
            print(f"WandB Project: {config.wandb.project}")
            if config.wandb.run_name:
                print(f"WandB Run: {config.wandb.run_name}")
        print("=" * 60)
        print()
        
        # Start Transformer training
        train_transformer(
            game_mode=config.game.mode,
            num_iterations=config.training.num_iterations,
            num_steps=agent_cfg.num_steps,
            minibatch_size=agent_cfg.minibatch_size,
            update_epochs=agent_cfg.update_epochs,
            d_model=agent_cfg.d_model,
            nhead=agent_cfg.nhead,
            num_layers=agent_cfg.num_layers,
            dropout=agent_cfg.dropout,
            lr_start=agent_cfg.lr_start,
            lr_end=agent_cfg.lr_end,
            gamma=agent_cfg.gamma,
            gae_lambda=agent_cfg.gae_lambda,
            clip_coef=agent_cfg.clip_coef,
            ent_coef_start=agent_cfg.ent_coef_start,
            ent_coef_end=agent_cfg.ent_coef_end,
            vf_coef=agent_cfg.vf_coef,
            max_grad_norm=agent_cfg.max_grad_norm,
            save_dir=config.training.save_dir,
            device=device,
            eval_every=config.training.eval_every,
            eval_episodes=config.training.eval_episodes,
            use_wandb=config.wandb.enabled,
            wandb_project=config.wandb.project,
            wandb_run_name=config.wandb.run_name
        )

    elif config.agent_type == "rppo":
        agent_cfg = config.agent_config

        # Print configuration
        print("\n" + "=" * 60)
        print("RPPO Training Configuration")
        print("=" * 60)
        print(f"Game Mode: {config.game.mode}")
        print(f"Device: {device}")
        print(f"Episodes: {config.training.num_episodes}")
        print(f"Learning Rate: {agent_cfg.lr}")
        print(f"Gamma: {agent_cfg.gamma}")
        print(f"GAE Lambda: {agent_cfg.gae_lambda}")
        print(f"Clip Eps: {agent_cfg.clip_eps}")
        print(f"Value Coef: {agent_cfg.value_coef}")
        print(f"Entropy Coef: {agent_cfg.entropy_coef}")
        print(f"Hidden Size: {agent_cfg.hidden_size}")
        print(f"Batch Size: {agent_cfg.batch_size}")
        print(f"WandB Enabled: {config.wandb.enabled}")
        print(f"WandB Project: {config.wandb.project}")
        print(f"WandB Run Name: {config.wandb.run_name}")
        print("=" * 60 + "\n")

        # Call RPPO trainer
        train_rppo(
            num_episodes=config.training.num_episodes,
            eval_every=config.training.eval_every,
            eval_episodes=config.training.eval_episodes,
            save_dir=config.training.save_dir,
            device=device,
            use_wandb=config.wandb.enabled,
            wandb_project=config.wandb.project,
            wandb_run_name=config.wandb.run_name,
            lr=agent_cfg.lr,
            gamma=agent_cfg.gamma,
            gae_lambda=agent_cfg.gae_lambda,
            clip_eps=agent_cfg.clip_eps,
            k_epochs=agent_cfg.k_epochs,
            value_coef=agent_cfg.value_coef,
            entropy_coef=agent_cfg.entropy_coef,
            max_grad_norm=agent_cfg.max_grad_norm,
            hidden_size=agent_cfg.hidden_size,
            batch_size=agent_cfg.batch_size,
        )

    else:
        raise ValueError(f"Unknown agent type: {config.agent_type}. Must be 'drqn', 'transformer', or 'rppo'")


if __name__ == "__main__":
    main()

