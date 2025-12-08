import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pyspiel
from junqi_drl.game import junqi_standard
from junqi_drl.agents import RandomAgent


def main():
    # Create log directory
    log_dir = project_root / "logs" / "autoplay"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"autoplay_results_{timestamp}.txt"
    
    # Create game
    game = pyspiel.load_game("junqi_standard")
    
    # Create two random agents
    agent0 = RandomAgent(player_id=0)
    agent1 = RandomAgent(player_id=1)
    agents = [agent0, agent1]
    
    # Play a game
    state = game.new_initial_state()
    move_count = 0
    
    # Open log file for writing
    with open(log_file, 'w') as f:
        # Write header
        f.write(f"Game: {game.get_type().long_name}\n")
        f.write(f"Board: {12}×{5}\n")
        f.write(f"Max game length: {800}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Write initial board
        f.write("Initial position:\n")
        f.write(str(state) + "\n\n")
        f.write("Starting game between two random agents...\n")
        f.write("=" * 70 + "\n\n")
        
        while not state.is_terminal():
            current_player = state.current_player()
            agent = agents[current_player]
            
            # Get action from agent
            action = agent.step(state)
            
            if action is None:
                f.write(f"Player {current_player} has no legal moves!\n")
                break
            
            # Apply action
            action_str = state.action_to_string(current_player, action)
            
            # Only log actual moves (not piece selection)
            if hasattr(state, 'selecting_piece'):
                if not state.selecting_piece:  # This is a move action
                    move_count += 1
                    f.write(f"Move {move_count}: Player {current_player} - {action_str}\n")
            
            state.apply_action(action)
        
        # Write game over info
        f.write("\n" + "=" * 70 + "\n")
        f.write("GAME OVER!\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total moves: {move_count}\n\n")
        f.write("Final board:\n")
        f.write(str(state) + "\n\n")
        
        returns = state.returns()
        f.write(f"Final returns: {returns}\n")
        if returns[0] > 0:
            winner = "Player 0 (Yellow/Random Agent 0) wins!"
        elif returns[0] < 0:
            winner = "Player 1 (Green/Random Agent 1) wins!"
        else:
            winner = "It's a draw!"
        f.write(winner + "\n")
    
    # Print summary to console
    print("Game completed!")
    print(f"Total moves: {move_count}")
    print(winner)
    print(f"\nGame log saved to: {log_file}")


if __name__ == "__main__":
    main()
