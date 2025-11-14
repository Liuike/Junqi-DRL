# Custom Junqi Game Implementation

This directory contains custom Junqi game implementations for the Junqi-DRL project.

## Files

- `junqi_standard.py`: Full official Junqi board (12×5) with standard predetermined setup
- `__init__.py`: Package initialization

## Usage

```python
import pyspiel
from junqi_drl.game import junqi_standard

# Create game instance
game = pyspiel.load_game("junqi_standard")
state = game.new_initial_state()
```

Or run the example:
```bash
python junqi_drl/examples/junqi_standard_example.py
```

## Features

### `junqi_standard.py`
- **Full 12×5 official board**: 5 rows per player + 2 middle rows
- **Complete map elements**: Railways, camps, headquarters
- **Standard predetermined setup**: Competitive opening position (no deployment phase)
- **All 25 pieces per player**: Proper piece quantities according to official rules
- **Imperfect information**: Opponent pieces are hidden
- **Official combat rules**: Bombs, mines, engineers, flag capture

## OpenSpiel Integration

These implementations are registered with OpenSpiel and can be loaded using `pyspiel.load_game()`:
- `"junqi_standard"`: Standard board with predetermined setup

## Note

This is a custom implementation separate from the installed `open-spiel-junqi` package. 
It lives in the project directory to allow easy modification and experimentation.
