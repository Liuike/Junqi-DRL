import os
import sys
import argparse
import torch
import pyspiel

# Ensure repository root on path.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from junqi_drl.agents.transformer_agent import TransformerAgent
from junqi_drl.agents.random_agent import RandomAgent
from junqi_drl.game import junqi_8x3  # noqa: F401
from junqi_drl.game import junqi_standard  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer agent.")
    parser.add_argument("--gamemode", default="junqi_8x3", choices=["junqi_8x3", "junqi_standard"])
    parser.add_argument("--model", required=True, help="Path to transformer .pth checkpoint")
    parser.add_argument("--games", type=int, default=200, help="Number of evaluation games")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opponent", choices=["random", "self"], default="random",
                        help="Opponent type: random or transformer self-play")
    return parser.parse_args()


def evaluate(model_path, gamemode, board_variant, num_games, device, opponent_type):
    game = pyspiel.load_game(gamemode)

    player_agent = TransformerAgent(
        player_id=0,
        board_variant=board_variant,
        model_path=model_path,
        device=device,
        deterministic=True,
    )

    if opponent_type == "random":
        opponent_agent = RandomAgent(player_id=1)
    else:
        opponent_agent = TransformerAgent(
            player_id=1,
            board_variant=board_variant,
            model_path=model_path,
            device=device,
            deterministic=True,
        )

    wins = draws = losses = 0

    for game_idx in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            current_player = state.current_player()
            if current_player == player_agent.player_id:
                action = player_agent.choose_action(state)
            else:
                if opponent_type == "random":
                    action = opponent_agent.step(state)
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

        if (game_idx + 1) % max(1, num_games // 10) == 0:
            print(
                f"Completed {game_idx + 1}/{num_games} games | "
                f"WR {wins/(game_idx+1):.2%}"
            )

    print("\n=== Evaluation Summary ===")
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"Win Rate:  {wins / num_games:.2%} ({wins})")
    print(f"Draw Rate: {draws / num_games:.2%} ({draws})")
    print(f"Loss Rate: {losses / num_games:.2%} ({losses})")


if __name__ == "__main__":
    args = parse_args()
    board_variant = "small" if args.gamemode == "junqi_8x3" else "standard"
    evaluate(
        model_path=args.model,
        gamemode=args.gamemode,
        board_variant=board_variant,
        num_games=args.games,
        device=args.device,
        opponent_type=args.opponent,
    )
