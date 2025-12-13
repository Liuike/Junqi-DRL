import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pyspiel
from junqi_drl.game import junqi_standard


def main():
    # Create a Junqi game instance with standard setup
    game = pyspiel.load_game("junqi_standard")
    
    print(f"Game: {game.get_type().long_name}")
    print(f"Board: {12}×{5} (full official size)")
    print(f"Number of players: {game.num_players()}")
    print(f"Number of distinct actions: {game.num_distinct_actions()}")
    print()
    
    # Create an initial state
    state = game.new_initial_state()
    
    print("Initial board with standard opening position:")
    print(state)
    print()
    print("=" * 70)
    print("LEGEND:")
    print("  Map Types:")
    print("    ═ = Railway (allows long-distance movement)")
    print("    △ = Camp (defensive position)")
    print("    ◆ = Headquarters (protected back row)")
    print()
    print("  Pieces:")
    print("    司=Commander(10)  军=Corps(9)      师=Division(8)  旅=Brigade(7)")
    print("    团=Regiment(6)   营=Battalion(5)  连=Company(4)   排=Platoon(3)")
    print("    工=Engineer(2)   炸=Bomb(12)      雷=Mine(11)     旗=Flag(1)")
    print()
    print("  Colors: Yellow=Player 0  Green=Player 1")
    print("=" * 70)
    print()
    
    # Play game with human input
    print("Playing Junqi - enter action numbers to make moves...")
    print("(Enter 'q' to quit)\n")
    move_count = 0
    
    while not state.is_terminal():
        current_player = state.current_player()
        legal_actions = state.legal_actions()
        
        if not legal_actions:
            break
        
        print(f"\n{'='*70}")
        print(f"Move {move_count + 1} - Player {current_player}'s turn")
        if hasattr(state, 'selecting_piece') and state.selecting_piece:
            print("Phase: Select a piece to move")
        else:
            print(f"Phase: Select destination for piece at {state.selected_pos[current_player]}")
        print(f"{'='*70}")
        print(state)
        print()
        
        # Show available actions
        print(f"Legal actions ({len(legal_actions)} available):")
        num_to_show = min(30, len(legal_actions))
        for i, action in enumerate(legal_actions[:num_to_show]):
            action_str = state.action_to_string(current_player, action)
            print(f"  [{action:3d}] {action_str}", end="")
            if (i + 1) % 5 == 0:
                print()
            else:
                print("\t", end="")
        
        if len(legal_actions) > num_to_show:
            print(f"\n  ... and {len(legal_actions) - num_to_show} more actions")
        else:
            print()
        
        # Get user input
        while True:
            user_input = input(f"\nEnter action number (or 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                print("Game quit by user.")
                return
            
            try:
                action = int(user_input)
                if action in legal_actions:
                    break
                else:
                    print(f"Invalid action. Must be one of the legal actions.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")
        
        # Apply the action
        state.apply_action(action)
        
        # Only increment move count after a complete move (piece selected and moved)
        if hasattr(state, 'selecting_piece') and state.selecting_piece:
            move_count += 1
    
    print("\n" + "="*70)
    print("GAME OVER!")
    print("="*70)
    print("Final board:")
    print(state)
    print()
    
    if state.is_terminal():
        returns = state.returns()
        print(f"Final returns: {returns}")
        if returns[0] > 0:
            print("Player 0 (Yellow) wins!")
        elif returns[0] < 0:
            print("Player 1 (Green) wins!")
        else:
            print("It's a draw!")
    else:
        print(f"Game incomplete after {move_count} moves")


if __name__ == "__main__":
    main()
