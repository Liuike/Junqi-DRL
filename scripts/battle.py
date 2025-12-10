#!/usr/bin/env python3
"""
Quick head-to-head battle script for testing individual model matchups.

Usage examples:
    python scripts/battle.py drql vs transformer --num_games 50
    python scripts/battle.py rppo vs drql --num_games 100 --verbose
    python scripts/battle.py transformer vs random --num_games 20
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pyspiel

# Import game modules to register them with PySpiel
from junqi_drl.game import junqi_8x3
from junqi_drl.game import junqi_standard

from junqi_drl.agents.drql import DRQLAgent
from junqi_drl.agents.rppo import RPPoAgent
from junqi_drl.agents.transformer_agent import TransformerAgent
from junqi_drl.agents.random_agent import RandomAgent


def load_agent(agent_type: str, player_id: int, game_mode: str, device: str):
    """Load an agent by type."""
    final_models_dir = project_root / "final_models"
    board_variant = "small" if game_mode == "junqi_8x3" else "standard"
    
    if agent_type.lower() == "drql":
        model_path = final_models_dir / "drql_spatial_final.pth"
        game = pyspiel.load_game(game_mode)
        action_dim = game.num_distinct_actions()
        agent = DRQLAgent(
            player_id=player_id,
            action_dim=action_dim,
            game_mode=game_mode,
            network_type="spatial",
            device=device,
            hidden_size=256
        )
        agent.load(str(model_path))
        agent.epsilon = 0.0
        return agent, "DRQL"
    
    elif agent_type.lower() == "rppo":
        model_path = final_models_dir / "rppo_best.pt"
        game = pyspiel.load_game(game_mode)
        obs_dim = int(np.prod(game.observation_tensor_shape()))
        action_dim = game.num_distinct_actions()
        agent = RPPoAgent(
            player_id=player_id,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            hidden_size=256
        )
        agent.load(str(model_path))
        return agent, "RPPO"
    
    elif agent_type.lower() == "transformer":
        model_path = final_models_dir / "transformer_final.pth"
        agent = TransformerAgent(
            player_id=player_id,
            board_variant=board_variant,
            model_path=str(model_path),
            device=device,
            deterministic=True,
            d_model=128,
            nhead=8,
            num_layers=3,
            dropout=0.1
        )
        return agent, "Transformer"
    
    elif agent_type.lower() == "random":
        return RandomAgent(player_id=player_id), "Random"
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use: drql, rppo, transformer, or random")


def play_game(agent0, agent1, game, verbose: bool = False) -> int:
    """Play a single game. Returns 0 if agent0 wins, 1 if agent1 wins, -1 for draw."""
    state = game.new_initial_state()
    
    # Reset hidden states
    if hasattr(agent0, 'reset_hidden'):
        agent0.reset_hidden()
    if hasattr(agent1, 'reset_hidden'):
        agent1.reset_hidden()
    
    move_count = 0
    agents = [agent0, agent1]
    
    while not state.is_terminal():
        current_player = state.current_player()
        # current_player() may set terminal if no legal moves
        if state.is_terminal():
            break
        
        agent = agents[current_player]
        legal_actions = state.legal_actions(current_player)
        
        # Get action
        if hasattr(agent, 'choose_action'):
            action = agent.choose_action(state, legal_actions, eval_mode=True)
        else:
            # Fallback for agents without choose_action
            action = agent.choose_action(state)
        
        state.apply_action(action)
        move_count += 1
        
        if verbose and move_count % 50 == 0:
            print(f"  Move {move_count}...")
    
    returns = state.returns()
    winner = 0 if returns[0] > 0 else (1 if returns[0] < 0 else -1)
    
    if verbose:
        print(f"  Game completed in {move_count} moves")
    
    return winner


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick head-to-head battle between models")
    parser.add_argument("agent0", type=str, choices=["drql", "rppo", "transformer", "random"],
                        help="First agent type")
    parser.add_argument("vs", type=str, help="Literal 'vs' (ignored)")
    parser.add_argument("agent1", type=str, choices=["drql", "rppo", "transformer", "random"],
                        help="Second agent type")
    parser.add_argument("--num_games", type=int, default=50,
                        help="Number of games to play")
    parser.add_argument("--game_mode", type=str, default="junqi_8x3",
                        choices=["junqi_8x3", "junqi_standard"],
                        help="Game variant")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, or auto")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--swap", action="store_true",
                        help="Also run with swapped player positions")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load game
    game = pyspiel.load_game(args.game_mode)
    
    # Load agents
    print("Loading agents...")
    agent0, name0 = load_agent(args.agent0, 0, args.game_mode, device)
    agent1, name1 = load_agent(args.agent1, 1, args.game_mode, device)
    
    print("\n" + "="*60)
    print(f"BATTLE: {name0} vs {name1}")
    print("="*60)
    print(f"Game Mode: {args.game_mode}")
    print(f"Device: {device}")
    print(f"Number of games: {args.num_games}")
    print("="*60 + "\n")
    
    # Run games
    wins0 = 0
    wins1 = 0
    draws = 0
    
    for i in range(args.num_games):
        if args.verbose:
            print(f"Game {i+1}/{args.num_games}:")
        
        winner = play_game(agent0, agent1, game, args.verbose)
        
        if winner == 0:
            wins0 += 1
        elif winner == 1:
            wins1 += 1
        else:
            draws += 1
        
        if not args.verbose and (i + 1) % 10 == 0:
            current_wr0 = wins0 / (i + 1)
            print(f"Progress: {i+1}/{args.num_games} | {name0} WR: {current_wr0:.1%}")
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{name0} (Player 0): {wins0} wins ({wins0/args.num_games:.1%})")
    print(f"{name1} (Player 1): {wins1} wins ({wins1/args.num_games:.1%})")
    print(f"Draws: {draws} ({draws/args.num_games:.1%})")
    print("="*60)
    
    # Run swapped if requested
    if args.swap:
        print(f"\n{'='*60}")
        print(f"SWAPPED BATTLE: {name1} vs {name0}")
        print(f"{'='*60}\n")
        
        # Reload agents with swapped IDs
        agent1_swap, _ = load_agent(args.agent1, 0, args.game_mode, device)
        agent0_swap, _ = load_agent(args.agent0, 1, args.game_mode, device)
        
        wins1_swap = 0
        wins0_swap = 0
        draws_swap = 0
        
        for i in range(args.num_games):
            winner = play_game(agent1_swap, agent0_swap, game, False)
            
            if winner == 0:
                wins1_swap += 1
            elif winner == 1:
                wins0_swap += 1
            else:
                draws_swap += 1
            
            if (i + 1) % 10 == 0:
                current_wr1 = wins1_swap / (i + 1)
                print(f"Progress: {i+1}/{args.num_games} | {name1} WR: {current_wr1:.1%}")
        
        print("\n" + "="*60)
        print("SWAPPED RESULTS")
        print("="*60)
        print(f"{name1} (Player 0): {wins1_swap} wins ({wins1_swap/args.num_games:.1%})")
        print(f"{name0} (Player 1): {wins0_swap} wins ({wins0_swap/args.num_games:.1%})")
        print(f"Draws: {draws_swap} ({draws_swap/args.num_games:.1%})")
        print("="*60)
        
        # Combined stats
        total_games = args.num_games * 2
        total_wins0 = wins0 + wins0_swap
        total_wins1 = wins1 + wins1_swap
        total_draws = draws + draws_swap
        
        print("\n" + "="*60)
        print("COMBINED RESULTS (both positions)")
        print("="*60)
        print(f"{name0}: {total_wins0} wins ({total_wins0/total_games:.1%})")
        print(f"{name1}: {total_wins1} wins ({total_wins1/total_games:.1%})")
        print(f"Draws: {total_draws} ({total_draws/total_games:.1%})")
        print("="*60)


if __name__ == "__main__":
    main()
