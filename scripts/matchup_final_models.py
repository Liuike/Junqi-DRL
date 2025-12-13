#!/usr/bin/env python3
"""
Script to run matchups between all final models in a tournament format.

This script loads the three trained models from final_models/ directory:
- DRQL (Deep Recurrent Q-Learning)
- RPPO (Recurrent Proximal Policy Optimization)
- Transformer (with PPO)

And runs them against each other in all possible pairings to determine
the strongest model.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

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


class MatchupResults:
    """Track results of model matchups."""
    
    def __init__(self):
        self.results = {}
        self.total_games = {}
        
    def record_game(self, player0_name: str, player1_name: str, winner: int):
        """
        Record a game result.
        
        Args:
            player0_name: Name of player 0's model
            player1_name: Name of player 1's model
            winner: 0 if player0 won, 1 if player1 won, -1 for draw
        """
        matchup = (player0_name, player1_name)
        if matchup not in self.results:
            self.results[matchup] = {'wins': 0, 'losses': 0, 'draws': 0}
            self.total_games[matchup] = 0
            
        if winner == 0:
            self.results[matchup]['wins'] += 1
        elif winner == 1:
            self.results[matchup]['losses'] += 1
        else:
            self.results[matchup]['draws'] += 1
        
        self.total_games[matchup] += 1
    
    def get_stats(self, player0_name: str, player1_name: str) -> Dict:
        """Get statistics for a specific matchup."""
        matchup = (player0_name, player1_name)
        if matchup not in self.results:
            return None
        
        stats = self.results[matchup].copy()
        total = self.total_games[matchup]
        stats['total'] = total
        stats['win_rate'] = stats['wins'] / total if total > 0 else 0
        stats['draw_rate'] = stats['draws'] / total if total > 0 else 0
        stats['loss_rate'] = stats['losses'] / total if total > 0 else 0
        
        return stats
    
    def print_summary(self):
        """Print a summary of all matchup results."""
        print("\n" + "="*80)
        print("MATCHUP RESULTS SUMMARY")
        print("="*80)
        
        for (player0, player1), total in self.total_games.items():
            stats = self.get_stats(player0, player1)
            print(f"\n{player0} vs {player1} ({total} games)")
            print(f"  {player0} wins: {stats['wins']} ({stats['win_rate']:.1%})")
            print(f"  {player1} wins: {stats['losses']} ({stats['loss_rate']:.1%})")
            print(f"  Draws: {stats['draws']} ({stats['draw_rate']:.1%})")
        
        print("\n" + "="*80)


def load_drql_agent(model_path: str, game_mode: str, player_id: int, device: str) -> DRQLAgent:
    """Load DRQL agent from checkpoint."""
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
    
    agent.load(model_path)
    agent.epsilon = 0.0  
    
    return agent


def load_rppo_agent(model_path: str, game_mode: str, player_id: int, device: str) -> RPPoAgent:
    """Load RPPO agent from checkpoint."""
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
    
    agent.load(model_path)
    
    return agent


def load_transformer_agent(model_path: str, board_variant: str, player_id: int, device: str) -> TransformerAgent:
    """Load Transformer agent from checkpoint."""
    agent = TransformerAgent(
        player_id=player_id,
        board_variant=board_variant,
        model_path=model_path,
        device=device,
        deterministic=True,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1
    )
    
    return agent


def play_game(agent0, agent1, game, agent0_name: str, agent1_name: str, verbose: bool = False) -> int:
    """
    Play a single game between two agents.
    
    Args:
        agent0: Agent playing as player 0
        agent1: Agent playing as player 1
        game: PySpiel game object
        agent0_name: Name of agent 0 for logging
        agent1_name: Name of agent 1 for logging
        verbose: Print move-by-move details
        
    Returns:
        Winner: 0 if agent0 wins, 1 if agent1 wins, -1 for draw
    """
    state = game.new_initial_state()
    
    # Reset hidden states for recurrent agents
    if hasattr(agent0, 'reset_hidden'):
        agent0.reset_hidden()
    if hasattr(agent1, 'reset_hidden'):
        agent1.reset_hidden()
    
    move_count = 0
    agents = [agent0, agent1]
    
    while not state.is_terminal():
        current_player = state.current_player()
        if state.is_terminal():
            break
        
        agent = agents[current_player]
        legal_actions = state.legal_actions(current_player)
        
        action = agent.choose_action(state, legal_actions, eval_mode=True)

        if verbose and move_count % 50 == 0:
            print(f"  Move {move_count}: Player {current_player} ({['agent0', 'agent1'][current_player]})")
        
        state.apply_action(action)
        move_count += 1
    
    returns = state.returns()
    if returns[0] > 0:
        winner = 0
    elif returns[0] < 0:
        winner = 1
    else:
        winner = -1
    
    if verbose:
        winner_name = agent0_name if winner == 0 else (agent1_name if winner == 1 else "Draw")
        print(f"  Game over after {move_count} moves. Winner: {winner_name}")
    
    return winner


def run_matchup(agent0, agent1, agent0_name: str, agent1_name: str, 
                game, num_games: int, verbose: bool = True) -> Tuple[int, int, int]:
    """
    Run a matchup between two agents.
    
    Returns:
        (wins, losses, draws) from agent0's perspective
    """
    print(f"\nRunning {agent0_name} vs {agent1_name} ({num_games} games)...")
    
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(num_games):
        winner = play_game(agent0, agent1, game, agent0_name, agent1_name, verbose=False)
        
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1
        
        if verbose and (i + 1) % 20 == 0:
            current_wr = wins / (i + 1)
            print(f"  Progress: {i + 1}/{num_games} games | {agent0_name} WR: {current_wr:.1%}")
    
    win_rate = wins / num_games
    print(f"  Final: {agent0_name} WR: {win_rate:.1%} ({wins}W/{losses}L/{draws}D)")
    
    return wins, losses, draws


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run matchups between final models")
    parser.add_argument("--game_mode", type=str, default="junqi_8x3",
                        choices=["junqi_8x3", "junqi_standard"],
                        help="Game variant to use")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games per matchup")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, or auto")
    parser.add_argument("--include_random", action="store_true",
                        help="Include random agent in matchups")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")
    parser.add_argument("--save_log", action="store_true", default=True,
                        help="Save results to log file")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("FINAL MODELS MATCHUP")
    print("="*80)
    print(f"Game Mode: {args.game_mode}")
    print(f"Device: {device}")
    print(f"Games per matchup: {args.num_games}")
    print("="*80)
    
    # Setup paths
    final_models_dir = project_root / "final_models"
    drql_path = final_models_dir / "drql_best_wr0.960_ep2000.pth"
    rppo_path = final_models_dir / "rppo_best.pt"
    transformer_path = final_models_dir / "transformer_final.pth"
    
    # Verify all models exist
    for path, name in [(drql_path, "DRQL"), (rppo_path, "RPPO"), (transformer_path, "Transformer")]:
        if not path.exists():
            print(f"ERROR: {name} model not found at {path}")
            return
    
    # Load game
    game = pyspiel.load_game(args.game_mode)
    board_variant = "small" if args.game_mode == "junqi_8x3" else "standard"
    
    results = MatchupResults()
    
    models_config = [
        ("DRQL", lambda pid: load_drql_agent(str(drql_path), args.game_mode, pid, device)),
        ("RPPO", lambda pid: load_rppo_agent(str(rppo_path), args.game_mode, pid, device)),
        ("Transformer", lambda pid: load_transformer_agent(str(transformer_path), board_variant, pid, device)),
    ]
    
    if args.include_random:
        models_config.append(("Random", lambda pid: RandomAgent(player_id=pid)))
    
    # Run all pairwise matchups
    for i, (name0, loader0) in enumerate(models_config):
        for j, (name1, loader1) in enumerate(models_config):
            if i >= j: 
                continue
            
            print(f"\n{'='*80}")
            print(f"MATCHUP: {name0} vs {name1}")
            print(f"{'='*80}")
            
            # Load agents
            agent0 = loader0(0)
            agent1 = loader1(1)
            
            # Run matchup
            wins, losses, draws = run_matchup(
                agent0, agent1, name0, name1, game, args.num_games, args.verbose
            )
            
            # Record results
            for _ in range(wins):
                results.record_game(name0, name1, 0)
            for _ in range(losses):
                results.record_game(name0, name1, 1)
            for _ in range(draws):
                results.record_game(name0, name1, -1)
    
    # Print final summary
    results.print_summary()
    
    # Calculate overall rankings
    print("\n" + "="*80)
    print("OVERALL RANKINGS")
    print("="*80)
    
    model_names = [name for name, _ in models_config]
    scores = {name: 0 for name in model_names}
    
    for (p0, p1), total in results.total_games.items():
        stats = results.get_stats(p0, p1)
        # Score: 1 point per win, 0.5 per draw
        scores[p0] += stats['wins'] + 0.5 * stats['draws']
        scores[p1] += stats['losses'] + 0.5 * stats['draws']
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(ranked, 1):
        total_possible = args.num_games * (len(model_names) - 1)
        print(f"{rank}. {name}: {score:.1f} points ({score/total_possible:.1%} win rate)")
    
    # Save log file
    if args.save_log:
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"matchup_results_{timestamp}.txt"
        
        with open(log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FINAL MODELS MATCHUP RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Game Mode: {args.game_mode}\n")
            f.write(f"Games per matchup: {args.num_games}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for (p0, p1), total in results.total_games.items():
                stats = results.get_stats(p0, p1)
                f.write(f"{p0} vs {p1} ({total} games)\n")
                f.write(f"  {p0} wins: {stats['wins']} ({stats['win_rate']:.1%})\n")
                f.write(f"  {p1} wins: {stats['losses']} ({stats['loss_rate']:.1%})\n")
                f.write(f"  Draws: {stats['draws']} ({stats['draw_rate']:.1%})\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL RANKINGS\n")
            f.write("="*80 + "\n")
            for rank, (name, score) in enumerate(ranked, 1):
                total_possible = args.num_games * (len(model_names) - 1)
                f.write(f"{rank}. {name}: {score:.1f} points ({score/total_possible:.1%})\n")
        
        print(f"\nResults saved to: {log_file}")


if __name__ == "__main__":
    main()
