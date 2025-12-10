import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import pyspiel
from junqi_drl.game import junqi_8x3
from junqi_drl.game import junqi_standard
from junqi_drl.agents.drql import DRQLAgent
from junqi_drl.agents.transformer_agent import TransformerAgent
from junqi_drl.agents.random_agent import RandomAgent
from junqi_drl.core.metrics import MetricsLogger


def evaluate_drqn(
    model_path,
    game_mode="junqi_8x3",
    network_type="spatial",
    num_games=100,
    device="cpu",
    verbose=True,
    use_wandb=False,
    wandb_project="junqi-eval",
    wandb_run_name=None,
    hidden_size=256
):
    """
    Evaluate a trained DRQN model.
    
    Args:
        model_path: Path to model checkpoint
        game_mode: "junqi_8x3" or "junqi_standard"
        network_type: "flat" or "spatial"
        num_games: Number of evaluation games
        device: "cpu" or "cuda"
        verbose: Print detailed results
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        hidden_size: Hidden layer size (must match checkpoint)
    """
    # Initialize metrics logger
    model_name = os.path.basename(model_path)
    run_name = wandb_run_name or f"eval_{model_name}"
    
    config = {
        "agent_type": "drqn",
        "model_path": model_path,
        "game_mode": game_mode,
        "network_type": network_type,
        "num_games": num_games,
        "device": str(device),
        "opponent": "random",
        "hidden_size": hidden_size
    }
    
    metrics_logger = MetricsLogger(
        use_wandb=use_wandb,
        wandb_config={
            "project": wandb_project,
            "name": run_name,
            "config": config
        }
    )
        
    game = pyspiel.load_game(game_mode)
    action_dim = game.num_distinct_actions()

    # Create agent based on network type
    if network_type == "flat":
        obs_dim = int(np.prod(game.observation_tensor_shape()))
        drql_agent = DRQLAgent(
            player_id=0,
            obs_dim=obs_dim,
            action_dim=action_dim,
            network_type="flat",
            device=device,
            hidden_size=hidden_size
        )
    else:
        drql_agent = DRQLAgent(
            player_id=0,
            action_dim=action_dim,
            game_mode=game_mode,
            network_type=network_type,
            device=device,
            hidden_size=hidden_size
        )
    
    drql_agent.load(model_path)
    drql_agent.epsilon = 0.0  # Greedy evaluation
    
    # Random opponent
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
            # current_player() may set terminal if no legal moves
            if state.is_terminal():
                break
            
            if current_player == 0:
                legal_actions = state.legal_actions(0)
                action = drql_agent.choose_action(state, legal_actions, eval_mode=True)
                move_count += 1
            else:
                legal_actions = state.legal_actions(1)
                action = agent.choose_action(state, legal_actions, eval_mode=True)
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
    
    # Log metrics
    metrics_logger.log({
        "eval_vs_random/winrate": win_rate,
        "eval_vs_random/drawrate": draw_rate,
        "eval_vs_random/lossrate": loss_rate,
        "eval_vs_random/wins": wins,
        "eval_vs_random/draws": draws,
        "eval_vs_random/losses": losses,
        "eval_vs_random/avg_moves": avg_moves
    })
    metrics_logger.finish()
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'avg_moves': avg_moves
    }


def evaluate_transformer(
    model_path,
    game_mode="junqi_8x3",
    num_games=100,
    device="cpu",
    opponent_type="random",
    verbose=True,
    use_wandb=False,
    wandb_project="junqi-eval",
    wandb_run_name=None,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1
):
    """
    Evaluate a trained Transformer model.
    
    Args:
        model_path: Path to model checkpoint
        game_mode: "junqi_8x3" or "junqi_standard"
        num_games: Number of evaluation games
        device: "cpu" or "cuda"
        opponent_type: "random" or "self"
        verbose: Print detailed results
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
        d_model: Transformer model dimension (must match checkpoint)
        nhead: Number of attention heads (must match checkpoint)
        num_layers: Number of transformer layers (must match checkpoint)
        dropout: Dropout rate (must match checkpoint)
    """
    # Initialize metrics logger
    model_name = os.path.basename(model_path)
    run_name = wandb_run_name or f"eval_{model_name}"
    
    board_variant = "small" if game_mode == "junqi_8x3" else "standard"
    
    config = {
        "agent_type": "transformer",
        "model_path": model_path,
        "game_mode": game_mode,
        "board_variant": board_variant,
        "num_games": num_games,
        "opponent_type": opponent_type,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
        "device": str(device)
    }
    
    metrics_logger = MetricsLogger(
        use_wandb=use_wandb,
        wandb_config={
            "project": wandb_project,
            "name": run_name,
            "config": config
        }
    )
    
    game = pyspiel.load_game(game_mode)

    player_agent = TransformerAgent(
        player_id=0,
        board_variant=board_variant,
        model_path=model_path,
        device=device,
        deterministic=True,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )

    if opponent_type == "random":
        opponent_agent = RandomAgent(player_id=1)
    else:  # self-play
        opponent_agent = TransformerAgent(
            player_id=1,
            board_variant=board_variant,
            model_path=model_path,
            device=device,
            deterministic=True,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )

    wins = 0
    draws = 0
    losses = 0

    for i in range(num_games):
        state = game.new_initial_state()
        
        while not state.is_terminal():
            current_player = state.current_player()
            # current_player() may set terminal if no legal moves
            if state.is_terminal():
                break
            
            if current_player == player_agent.player_id:
                action = player_agent.choose_action(state)
            else:
                legal_actions = state.legal_actions(current_player)
                if opponent_type == "random":
                    action = opponent_agent.choose_action(state, legal_actions, eval_mode=True)
                else:
                    action = opponent_agent.choose_action(state)
            state.apply_action(action)

        result = state.returns()[player_agent.player_id]
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

        if verbose and (i + 1) % max(1, num_games // 10) == 0:
            current_wr = wins / (i + 1)
            print(f"  Completed {i + 1}/{num_games} games | WR {current_wr:.2%}")
            
            # Log intermediate metrics
            metrics_logger.log({
                "games_played": i + 1,
                "current_winrate": current_wr,
                "current_wins": wins,
                "current_draws": draws,
                "current_losses": losses
            }, step=i + 1)

    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games

    print(f"\n{'='*60}")
    print(f"Evaluation vs {opponent_type.capitalize()} ({num_games} games)")
    print(f"{'='*60}")
    print(f"Win Rate:       {win_rate:.2%} ({wins} wins)")
    print(f"Draw Rate:      {draw_rate:.2%} ({draws} draws)")
    print(f"Loss Rate:      {loss_rate:.2%} ({losses} losses)")
    print(f"{'='*60}\n")
    
    # Log final metrics
    metrics_logger.log({
        "eval/final_winrate": win_rate,
        "eval/final_wins": wins,
        "eval/final_draws": draws,
        "eval/final_losses": losses,
        "eval/total_games": num_games
    })
    
    metrics_logger.finish()
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DRQN or Transformer model")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--agent_type", type=str, required=True,
                        choices=["drqn", "transformer"],
                        help="Agent type: 'drqn' or 'transformer'")
    parser.add_argument("--game_mode", type=str, default="junqi_8x3", 
                        choices=["junqi_8x3", "junqi_standard"],
                        help="Game variant")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of evaluation games")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="junqi-eval",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")
    
    # DRQN-specific arguments
    parser.add_argument("--network_type", type=str, default="spatial",
                        choices=["flat", "spatial"],
                        help="[DRQN] Network architecture (must match checkpoint)")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="[DRQN] Hidden layer size (must match checkpoint)")
    
    # Transformer-specific arguments
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "self"],
                        help="[Transformer] Opponent type: 'random' or 'self'")
    parser.add_argument("--d_model", type=int, default=64,
                        help="[Transformer] Model dimension (must match checkpoint)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="[Transformer] Number of attention heads (must match checkpoint)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="[Transformer] Number of transformer layers (must match checkpoint)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="[Transformer] Dropout rate (must match checkpoint)")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print(f"{args.agent_type.upper()} Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Game Mode: {args.game_mode}")
    print(f"Device: {device}")
    print(f"Evaluation Games: {args.num_games}")
    
    if args.agent_type == "drqn":
        print(f"Network Type: {args.network_type}")
        print(f"Hidden Size: {args.hidden_size}")
        print("=" * 60)
        print()
        
        evaluate_drqn(
            model_path=args.model_path,
            game_mode=args.game_mode,
            network_type=args.network_type,
            num_games=args.num_games,
            device=device,
            verbose=args.verbose,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            hidden_size=args.hidden_size
        )
    else:  # transformer
        print(f"Opponent: {args.opponent}")
        print(f"Architecture: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")
        print("=" * 60)
        print()
        
        evaluate_transformer(
            model_path=args.model_path,
            game_mode=args.game_mode,
            num_games=args.num_games,
            device=device,
            opponent_type=args.opponent,
            verbose=args.verbose,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
