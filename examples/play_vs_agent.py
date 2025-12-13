import sys
from pathlib import Path

# Add the project root to the path
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


def load_agent(agent_type: str, board_variant: str, player_id: int, device: str):
    """Load the specified agent."""
    
    # Determine game mode and model path
    if board_variant == "small":
        game_mode = "junqi_8x3"
        if agent_type == "drql":
            model_path = "final_models/drql_spatial_final_8x3.pth"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"DRQL model for small board not found: {model_path}")
        elif agent_type == "rppo":
            model_path = "final_models/rppo_best_8x3.pt"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"RPPO model for small board not found: {model_path}")
        elif agent_type == "transformer":
            model_path = "final_models/transformer_final_8x3.pth"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Transformer model for small board not found: {model_path}")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    elif board_variant == "standard":
        game_mode = "junqi_standard"
        if agent_type == "drql":
            model_path = "final_models/drql_spatial_final_standard.pth"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"DRQL model for standard board not found: {model_path}")
        elif agent_type == "rppo":
            model_path = "final_models/rppo_best_standard.pt"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"RPPO model for standard board not found: {model_path}")
        elif agent_type == "transformer":
            model_path = "final_models/transformer_final_standard.pth"
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Transformer model for standard board not found: {model_path}")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    else:
        raise ValueError(f"Unknown board variant: {board_variant}")
    
    game = pyspiel.load_game(game_mode)
    
    if agent_type == "drql":
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
        agent.epsilon = 0.0  # Greedy evaluation
        
    elif agent_type == "rppo":
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
        
    elif agent_type == "transformer":
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
    
    return agent, game


def print_game_info(game, board_variant):
    """Print game information."""
    print("=" * 80)
    print("JUNQI - PLAY AGAINST AI")
    print("=" * 80)
    print(f"Game: {game.get_type().long_name}")
    if board_variant == "small":
        print(f"Board: 8×3 (simplified)")
    else:
        print(f"Board: 12×5 (standard)")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")
    print()
    print("LEGEND:")
    print("  Map Types:")
    print("    ═ = Railway (allows long-distance movement)")
    print("    △ = Camp (defensive position)")
    print("    ◆ = Headquarters (protected back row)")
    print()
    print("  Pieces:")
    print("    司=Commander(10)  军=Corps(9)      师=Division(8)  旅=Brigade(7)")
    if board_variant == "standard":
        print("    团=Regiment(6)   营=Battalion(5)  连=Company(4)   排=Platoon(3)")
    print("    工=Engineer(2)   炸=Bomb(12)      旗=Flag(1)")
    if board_variant == "standard":
        print("    雷=Mine(11)")
    print()
    print("  Colors: Yellow=Player 0  Green=Player 1")
    print("=" * 80)
    print()


def get_action_description(state, action, current_player):
    """Get a human-readable description of an action."""
    # Decode action to board position using state's decode_action list
    if hasattr(state, 'decode_action') and action < len(state.decode_action):
        pos = state.decode_action[action]
        row, col = pos[0], pos[1]
    else:
        # Fallback decoding
        num_cols = 3 if hasattr(state, 'board') and len(state.board[0]) == 3 else 5
        row = action // num_cols
        col = action % num_cols
    
    if hasattr(state, 'selecting_piece') and state.selecting_piece:
        # Selecting a piece
        if hasattr(state, 'board') and row < len(state.board) and col < len(state.board[0]):
            piece = state.board[row][col]
            if piece.country == current_player:
                piece_name = piece.name
                return f"Select piece at ({row},{col}): {piece_name}"
        return f"Select ({row},{col})"
    else:
        # Moving to destination
        return f"Move to ({row},{col})"


def play_game(agent, game, human_player: int, agent_name: str, board_variant: str):
    """
    Play an interactive game against the agent.
    
    Args:
        agent: The AI agent to play against
        game: PySpiel game object
        human_player: 0 or 1 (which player is human)
        agent_name: Name of the agent for display
        board_variant: "small" or "standard"
    """
    state = game.new_initial_state()
    
    # Reset agent hidden state if applicable
    if hasattr(agent, 'reset_hidden'):
        agent.reset_hidden()
    
    print_game_info(game, board_variant)
    print(f"You are Player {human_player}")
    print(f"AI ({agent_name}) is Player {1 - human_player}")
    print()
    print("Starting position:")
    print(state)
    print()
    
    move_count = 0
    
    while not state.is_terminal():
        current_player = state.current_player()
        if state.is_terminal():
            break
        legal_actions = state.legal_actions(current_player)
        
        if not legal_actions:
            print("No legal actions available!")
            break
        
        print(f"\n{'='*80}")
        print(f"Move {move_count + 1} - ", end="")
        
        if current_player == human_player:
            print("YOUR TURN")
            if hasattr(state, 'selecting_piece') and state.selecting_piece:
                print("Phase: Select a piece to move")
            else:
                selected = state.selected_pos[current_player] if hasattr(state, 'selected_pos') else None
                if selected:
                    print(f"Phase: Select destination for piece at ({selected[0]},{selected[1]})")
        else:
            print(f"AI TURN ({agent_name})")
        
        print(f"{'='*80}")
        print(state)
        print()
        
        if current_player == human_player:
            # Human player's turn
            print(f"Legal actions ({len(legal_actions)} available):")
            num_to_show = min(40, len(legal_actions))
            
            for i, action in enumerate(legal_actions[:num_to_show]):
                action_desc = get_action_description(state, action, current_player)
                print(f"  [{action:3d}] {action_desc}", end="")
                if (i + 1) % 4 == 0:
                    print()
                else:
                    print("\t", end="")
            
            if len(legal_actions) > num_to_show:
                print(f"\n  ... and {len(legal_actions) - num_to_show} more actions")
            else:
                if num_to_show % 4 != 0:
                    print()
            
            # Get user input
            while True:
                user_input = input(f"\nEnter action number (or 'q' to quit, 'h' for help): ").strip()
                
                if user_input.lower() == 'q':
                    print("Game quit by user.")
                    return None
                
                if user_input.lower() == 'h':
                    print("\nHelp:")
                    print("  - Enter the number of the action you want to take")
                    print("  - First, select a piece to move")
                    print("  - Then, select where to move it")
                    print("  - 'q' to quit")
                    continue
                
                try:
                    action = int(user_input)
                    if action in legal_actions:
                        break
                    else:
                        print(f"Invalid action. Must be one of: {legal_actions[:20]}{'...' if len(legal_actions) > 20 else ''}")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"\nYou chose: {get_action_description(state, action, current_player)}")
        
        else:
            # AI agent's turn
            if hasattr(agent, 'choose_action'):
                # DRQL or RPPO agent
                action = agent.choose_action(state, legal_actions, eval_mode=True)
            else:
                # Transformer agent
                action = agent.choose_action(state)
            
            action_desc = get_action_description(state, action, current_player)
            print(f"AI chooses: [{action}] {action_desc}")
            input("Press Enter to continue...")
        
        state.apply_action(action)
        move_count += 1
    
    # Game over
    print("\n" + "=" * 80)
    print("GAME OVER!")
    print("=" * 80)
    print(state)
    print()
    
    returns = state.returns()
    if returns[human_player] > 0:
        print("🎉 YOU WIN! 🎉")
    elif returns[human_player] < 0:
        print("💔 You lost. Better luck next time!")
    else:
        print("🤝 It's a DRAW!")
    
    print(f"\nTotal moves: {move_count}")
    print(f"Final returns: Player 0: {returns[0]:.1f}, Player 1: {returns[1]:.1f}")
    
    return returns[human_player]


def main():
    """Main function to run the interactive game."""
    
    print("Welcome to Junqi - Play Against AI!")
    print()
    
    # Choose board size
    while True:
        board_input = input("Choose board size (1=Small 8x3, 2=Standard 12x5) [1]: ").strip()
        if board_input == "" or board_input == "1":
            board_variant = "small"
            break
        elif board_input == "2":
            board_variant = "standard"
            break
        else:
            print("Please enter 1 or 2")
    
    # Choose opponent
    print("\nChoose your opponent:")
    print("  1. DRQL (Deep Recurrent Q-Learning)")
    print("  2. RPPO (Recurrent Proximal Policy Optimization)")
    print("  3. Transformer (Attention-based model)")
    
    while True:
        agent_input = input("Enter choice [1]: ").strip()
        if agent_input == "" or agent_input == "1":
            agent_type = "drql"
            agent_name = "DRQL"
            break
        elif agent_input == "2":
            agent_type = "rppo"
            agent_name = "RPPO"
            break
        elif agent_input == "3":
            agent_type = "transformer"
            agent_name = "Transformer"
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Choose side
    while True:
        side_input = input("\nChoose your side (0=First/Yellow, 1=Second/Green) [0]: ").strip()
        if side_input == "" or side_input == "0":
            human_player = 0
            break
        elif side_input == "1":
            human_player = 1
            break
        else:
            print("Please enter 0 or 1")
    
    # Load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {agent_name} model (device: {device})...")
    
    try:
        agent, game = load_agent(agent_type, board_variant, 1 - human_player, device)
        print("Model loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"Error: Could not find model file. {e}")
        print("Make sure you're running this from the project root directory.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Play the game
    try:
        result = play_game(agent, game, human_player, agent_name, board_variant)
        
        if result is not None:
            # Ask if they want to play again
            play_again = input("\nPlay again? (y/n) [n]: ").strip().lower()
            if play_again == 'y':
                main()
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError during game: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
