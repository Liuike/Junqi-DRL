import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import pyspiel
from junqi_drl.game import junqi_8x3
from junqi_drl.game import junqi_standard
from junqi_drl.agents.drql import DRQLAgent
from junqi_drl.agents.random_agent import RandomAgent

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, metrics will not be logged to W&B")

gamemode = "junqi_8x3"


def evaluate_model(model_path, num_games=100, device="cpu", verbose=True, use_wandb=False, wandb_project="junqi-drql-eval", wandb_run_name=None):
    # Initialize wandb if requested
    if use_wandb and WANDB_AVAILABLE:
        model_name = os.path.basename(model_path)
        run_name = wandb_run_name or f"eval_{model_name}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model_path": model_path,
                "num_games": num_games,
                "device": str(device),
                "opponent": "random"
            }
        )
        use_wandb = True
    else:
        use_wandb = False
        
    game = pyspiel.load_game(gamemode)
    obs_dim = np.prod(game.observation_tensor_shape())
    action_dim = game.num_distinct_actions()

    drql_agent = DRQLAgent(
        player_id=0, 
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        device=device
    )
    drql_agent.load(model_path)
    drql_agent.epsilon = 0.0 
    
    #AGENT
    agent = RandomAgent(player_id=1)

    wins = 0
    draws = 0
    losses = 0
    total_moves = []

    for i in range(num_games):
        state = game.new_initial_state()
        drql_agent.reset_hidden()
        move_count = 0

        while not state.is_terminal():
            current_player = state.current_player()
            if current_player == 0:
                legal_actions = state.legal_actions(0)
                action = drql_agent.choose_action(state, legal_actions, eval_mode=True)
                move_count += 1
            else:
                action = agent.step(state)
            state.apply_action(action)

        returns = state.returns()
        total_moves.append(move_count)
        
        if returns[0] > 0:
            wins += 1
        elif returns[0] == 0:
            draws += 1
        else:
            losses += 1

        if verbose and (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{num_games} games...")

    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    avg_moves = np.mean(total_moves)
    
    print(f"\n{'='*60}")
    print(f"Evaluation vs Random ({num_games} games)")
    print(f"{'='*60}")
    print(f"Win Rate:       {win_rate:.2%} ({wins} wins)")
    print(f"Draw Rate:      {draw_rate:.2%} ({draws} draws)")
    print(f"Loss Rate:      {loss_rate:.2%} ({losses} losses)")
    print(f"Avg Moves:      {avg_moves:.1f}")
    print(f"{'='*60}\n")
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            "eval_vs_random/winrate": win_rate,
            "eval_vs_random/drawrate": draw_rate,
            "eval_vs_random/lossrate": loss_rate,
            "eval_vs_random/wins": wins,
            "eval_vs_random/draws": draws,
            "eval_vs_random/losses": losses,
            "eval_vs_random/avg_moves": avg_moves
        })
        wandb.finish()
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'avg_moves': avg_moves
    }
