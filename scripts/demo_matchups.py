#!/usr/bin/env python3
"""
Quick demo script to showcase all three final models.
Runs short matchups between all models to demonstrate their capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess


def run_command(cmd):
    """Run a command and display output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    subprocess.run(cmd)


def main():
    scripts_dir = project_root / "scripts"
    battle_script = scripts_dir / "battle.py"
    
    print("\n" + "="*80)
    print("FINAL MODELS DEMONSTRATION")
    print("="*80)
    print("\nThis demo will run quick matchups between all three final models:")
    print("  1. DRQL (Deep Recurrent Q-Learning)")
    print("  2. RPPO (Recurrent Proximal Policy Optimization)")
    print("  3. Transformer (with PPO)")
    print("\nEach matchup will run 20 games to give you a quick preview.")
    print("For comprehensive evaluation, use the full tournament script.")
    print("="*80)
    
    input("\nPress Enter to start the demo...")
    
    matchups = [
        ("drql", "rppo", "DRQL vs RPPO"),
        ("drql", "transformer", "DRQL vs Transformer"),
        ("rppo", "transformer", "RPPO vs Transformer"),
    ]
    
    for agent0, agent1, description in matchups:
        print(f"\n\n{'#'*80}")
        print(f"# MATCHUP: {description}")
        print(f"{'#'*80}\n")
        
        cmd = [
            "python", str(battle_script),
            agent0, "vs", agent1,
            "--num_games", "20"
        ]
        
        run_command(cmd)
        
        input(f"\nPress Enter to continue to next matchup...")
    
    print("\n\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nTo run more comprehensive evaluations:")
    print("  - Head-to-head: python scripts/battle.py MODEL1 vs MODEL2 --num_games 100 --swap")
    print("  - Full tournament: python scripts/matchup_final_models.py --num_games 100")
    print("  - See scripts/MATCHUP_README.md for detailed documentation")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
