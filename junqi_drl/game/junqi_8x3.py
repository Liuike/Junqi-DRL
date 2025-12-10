
"""
This is a custom implementation for the Junqi-DRL project. Smaller board for faster training.
"""
import copy
import enum

import numpy as np
import pyspiel

_NUM_PLAYERS = 2
_NUM_MAX_PEACE_STEP = 800  # Reduced for smaller board
_NUM_ROWS = 8   # 8 rows total: P0 top (0-3), P1 bottom (4-7)
_NUM_COLS = 3
_NUM_CELLS = _NUM_ROWS * _NUM_COLS
_NUM_CHESS_TYPES = 8  # Reduced types

# Simplified piece configuration (10 pieces per player)
_NUM_CHESS_QUANTITY = 10

_NUM_CHESS_QUANTITY_BY_TYPE = {
    12: 1,  # Bomb (炸弹)
    10: 1,  # Commander (司令)
    9: 1,   # Corps Commander (军长)
    8: 2,   # Division Commander (师长)
    7: 2,   # Brigade Commander (旅长)
    2: 2,   # Engineer (工兵)
    1: 1,   # Flag (军旗)
    # No mines to simplify learning
}

# Board layout constants
_RAILWAY_POSITIONS = [
    # Vertical railways in all 3 columns
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0],
    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1],
    [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2],
]

_CAMP_POSITIONS = [
    [1, 1], [2, 1],  # Player 0 camps
    [5, 1], [6, 1],  # Player 1 camps
]

_HQ_POSITIONS = [
    [0, 0], [0, 2],  # Player 0 HQs (top row)
    [7, 0], [7, 2],  # Player 1 HQs (bottom row)
]

# Standard simplified opening position (vertical layout)
_STANDARD_SETUP = {
    0: [  # Player 0 (top half: rows 0-3)
        [0, 0, 1],   # Flag in HQ
        [0, 1, 10],  # Commander
        [0, 2, 12],  # Bomb
        [1, 0, 8],   # Division
        [1, 1, 7],   # ← CHANGED from 4 to 7 (Brigade)
        [1, 2, 8],   # Division
        [2, 0, 7],   # Brigade
        [2, 1, 9],   # Corps
        [2, 2, 7],   # Brigade
        [3, 1, 2],   # Engineer
    ],
    1: [  # Player 1 (bottom half: rows 4-7)
        [4, 1, 2],   # Engineer
        [5, 0, 7],   # Brigade
        [5, 1, 9],   # Corps
        [5, 2, 7],   # Brigade
        [6, 0, 8],   # Division
        [6, 1, 7],   # ← CHANGED from 4 to 7 (Brigade)
        [6, 2, 8],   # Division
        [7, 0, 12],  # Bomb
        [7, 1, 10],  # Commander
        [7, 2, 1],   # Flag in HQ
    ]
}

# Since we removed some types, remap for display (keep 4,3 for compatibility, but don't use them)
_DICT_CHESS_NAME = {
    12: "炸", 10: "司", 9: "军",
    8: "师", 7: "旅", 2: "工", 1: "旗",
    0: "　", -10: "口"
}

_GAME_TYPE = pyspiel.GameType(
    short_name="junqi_8x3",
    long_name="Junqi 8x3 Simplified Board",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CELLS,
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=800
)


class ChessType(enum.IntEnum):
    """Enum chess type."""
    BOMB: int = 12
    COMMANDER: int = 10
    CORPS_COMMANDER: int = 9
    DIVISION_COMMANDER: int = 8
    BRIGADE_COMMANDER: int = 7
    ENGINEER: int = 2
    FLAG: int = 1
    NONE: int = 0
    UNKNOWN: int = -10


class MapType(enum.IntEnum):
    """Enum map type."""
    NORMAL: int = 0
    RAILWAY: int = 1
    CAMP: int = 2
    HQ: int = 3


class JunQi8x3Game(pyspiel.Game):
    """Junqi game with 8x3 simplified predetermined setup."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self, unusedarg=None):
        """Returns a state corresponding to the start of a game."""
        return JunQi8x3State(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return JunQi8x3Observer(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)


class JunQi8x3State(pyspiel.State):
    """A python version of the Junqi 8x3 state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)

        self._cur_player: int = 0
        self._player0_score: float = 0.0
        self._is_terminal: bool = False

        self.game_length: int = 0
        self.game_length_peace: int = 0
        self.draw: bool = False

        self.selected_pos = [None, None]
        self.flags_pos = [[0, 0], [7, 2]]  # Known flag positions
        self.decode_action = [[i // _NUM_COLS, i % _NUM_COLS] for i in range(_NUM_CELLS)]
        
        # Initialize board and map
        self.board = [[Chess(0, -1)] * _NUM_COLS for _ in range(_NUM_ROWS)]
        self.map = [[MapType.NORMAL] * _NUM_COLS for _ in range(_NUM_ROWS)]
        
        # Set up map types
        for pos in _RAILWAY_POSITIONS:
            if 0 <= pos[0] < _NUM_ROWS and 0 <= pos[1] < _NUM_COLS:
                self.map[pos[0]][pos[1]] = MapType.RAILWAY
        for pos in _CAMP_POSITIONS:
            if 0 <= pos[0] < _NUM_ROWS and 0 <= pos[1] < _NUM_COLS:
                self.map[pos[0]][pos[1]] = MapType.CAMP
        for pos in _HQ_POSITIONS:
            if 0 <= pos[0] < _NUM_ROWS and 0 <= pos[1] < _NUM_COLS:
                self.map[pos[0]][pos[1]] = MapType.HQ

        # Place pieces according to simplified setup
        for player in [0, 1]:
            for piece_info in _STANDARD_SETUP[player]:
                row, col, piece_type = piece_info
                if 0 <= row < _NUM_ROWS and 0 <= col < _NUM_COLS:
                    self.board[row][col] = Chess(piece_type, player)
        
        self.obs_mov = [[[0] * _NUM_COLS for _ in range(_NUM_ROWS)],
                        [[0] * _NUM_COLS for _ in range(_NUM_ROWS)]]
        self.obs_attack: bool = False
        self.selecting_piece: bool = True

    def current_player(self) -> int:
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        
        # Check if current player has no legal moves - they lose
        if not self._legal_actions(self._cur_player):
            self._is_terminal = True
            # Current player loses
            self._player0_score = -1.0 if self._cur_player == 0 else 1.0
            return pyspiel.PlayerId.TERMINAL
        
        return self._cur_player

    def _legal_actions(self, player: int):
        """Returns a list of legal actions."""
        actions = []
        
        if self.selecting_piece:
            # Selecting a piece to move
            for row in range(_NUM_ROWS):
                for col in range(_NUM_COLS):
                    if (self.board[row][col].country == player
                            and self.board[row][col].type != ChessType.FLAG
                            and self._has_legal_moves([row, col], player)):
                        actions.append(row * _NUM_COLS + col)
        else:
            # Selecting destination for selected piece
            from_pos = self.selected_pos[player]
            for to_pos in self._get_legal_destinations(from_pos, player):
                actions.append(to_pos[0] * _NUM_COLS + to_pos[1])
        
        # If no legal actions, player loses (handled in current_player)
        return actions

    def _has_legal_moves(self, from_pos, player: int) -> bool:
        """Check if a piece has any legal moves."""
        destinations = self._get_legal_destinations(from_pos, player)
        return len(destinations) > 0

    def _get_legal_destinations(self, from_pos, player: int):
        """Get all legal destination squares for a piece."""
        legal = []
        piece = self.board[from_pos[0]][from_pos[1]]
        
        # Basic adjacent moves
        legal.extend(self._get_adjacent_moves(from_pos, player))
        
        # Railway moves for all pieces except flag
        if (self.map[from_pos[0]][from_pos[1]] == MapType.RAILWAY
                and piece.type != ChessType.FLAG):
            legal.extend(self._get_railway_moves(from_pos, player))
        
        # Engineers can fly on railways
        if piece.type == ChessType.ENGINEER:
            legal.extend(self._get_engineer_railway_moves(from_pos, player))
        
        # Remove duplicates by converting to set of tuples and back
        unique_legal = []
        seen = set()
        for pos in legal:
            pos_tuple = tuple(pos)
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_legal.append(pos)
        
        return unique_legal

    def _get_adjacent_moves(self, from_pos, player: int):
        """Get legal adjacent square moves."""
        legal = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            to_r, to_c = from_pos[0] + dr, from_pos[1] + dc
            if (0 <= to_r < _NUM_ROWS and 0 <= to_c < _NUM_COLS
                    and self.board[to_r][to_c].country != player):
                legal.append([to_r, to_c])
        
        return legal

    def _get_railway_moves(self, from_pos, player: int):
        """Get moves along railway lines (straight lines only)."""
        legal = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            r, c = from_pos[0] + dr, from_pos[1] + dc
            while (0 <= r < _NUM_ROWS and 0 <= c < _NUM_COLS
                   and self.map[r][c] == MapType.RAILWAY):
                if self.board[r][c].type == ChessType.NONE:
                    legal.append([r, c])
                    r += dr
                    c += dc
                elif self.board[r][c].country != player:
                    legal.append([r, c])
                    break
                else:
                    break
        
        return legal

    def _get_engineer_railway_moves(self, from_pos, player: int):
        """Engineers can move to any connected railway position."""
        legal = []
        visited = set()
        queue = [from_pos]
        visited.add(tuple(from_pos))
        
        while queue:
            curr = queue.pop(0)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if (0 <= nr < _NUM_ROWS and 0 <= nc < _NUM_COLS
                        and (nr, nc) not in visited
                        and self.map[nr][nc] == MapType.RAILWAY):
                    visited.add((nr, nc))
                    if self.board[nr][nc].country == -1:
                        legal.append([nr, nc])
                        queue.append([nr, nc])
                    elif self.board[nr][nc].country != player:
                        legal.append([nr, nc])
        
        return legal

    def _apply_action(self, action: int) -> None:
        """Applies the specified action to the state."""
        player = self._cur_player

        # Check for draw conditions
        if (self.game_length >= _GAME_INFO.max_game_length
                or self.game_length_peace >= _NUM_MAX_PEACE_STEP):
            self._is_terminal = True
            self.draw = True
            return

        if self.selecting_piece:
            # Select piece
            self.selected_pos[player] = self.decode_action[action]
            self.selecting_piece = False
        else:
            # Move piece
            from_pos = self.selected_pos[player]
            to_pos = self.decode_action[action]
            
            self.obs_attack = False
            self.obs_mov = [[[0] * _NUM_COLS for _ in range(_NUM_ROWS)],
                           [[0] * _NUM_COLS for _ in range(_NUM_ROWS)]]
            
            attacker = self.board[from_pos[0]][from_pos[1]]
            defender = self.board[to_pos[0]][to_pos[1]]
            
            if defender.type == ChessType.NONE:
                # Move to empty square
                self.board[to_pos[0]][to_pos[1]] = copy.deepcopy(attacker)
                self.board[from_pos[0]][from_pos[1]] = Chess(0, -1)
                self.obs_mov[player][from_pos[0]][from_pos[1]] = -1
                self.obs_mov[1 - player][from_pos[0]][from_pos[1]] = -1
            else:
                # Combat
                self.obs_attack = True
                self.obs_mov[player][from_pos[0]][from_pos[1]] = -2
                self.obs_mov[1 - player][from_pos[0]][from_pos[1]] = -2
                
                winner = self._resolve_combat(attacker, defender)
                
                if winner == "attacker":
                    self.board[to_pos[0]][to_pos[1]] = copy.deepcopy(attacker)
                    if defender.type == ChessType.FLAG:
                        self._is_terminal = True
                        self._player0_score = 1.0 if player == 0 else -1.0
                elif winner == "defender":
                    pass  # Defender stays
                else:  # "both_die"
                    self.board[to_pos[0]][to_pos[1]] = Chess(0, -1)
                
                self.board[from_pos[0]][from_pos[1]] = Chess(0, -1)
            
            self.obs_mov[player][to_pos[0]][to_pos[1]] = 1
            self.obs_mov[1 - player][to_pos[0]][to_pos[1]] = 1
            
            # Update counters
            self.game_length_peace += 1
            if self.obs_attack:
                self.game_length_peace = 0
            
            # Switch player
            self._cur_player = 1 - self._cur_player
            self.selecting_piece = True
            self.game_length += 1

    def _resolve_combat(self, attacker, defender):
        """Resolve combat between two pieces."""
        # Special case: bomb
        if attacker.type == ChessType.BOMB or defender.type == ChessType.BOMB:
            return "both_die"
        
        # No mines in this variant

        if attacker.type == defender.type:
            return "both_die"
        
        # Normal combat - higher rank wins
        if attacker.type > defender.type:
            return "attacker"
        else:
            return "defender"

    def _action_to_string(self, player, action):
        """Action -> string."""
        pos = self.decode_action[action]
        return f"P{player}:({pos[0]},{pos[1]})"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        if self.draw:
            return [0.0, 0.0]
        return [self._player0_score, -self._player0_score]

    def __str__(self):
        """String for debug purposes."""
        return _board_to_string(self.board, self.map)


class JunQi8x3Observer:
    """Observer for Junqi 8x3 game."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        
        # Features: own pieces (8) + opponent pieces (8) + map types (4) + other info (3)
        self.shape = (_NUM_ROWS, _NUM_COLS, _NUM_CHESS_TYPES + _NUM_CHESS_TYPES + 4 + 3)
        self.tensor = np.zeros(np.prod(self.shape), np.float32)
        self.dict = {"observation": np.reshape(self.tensor, self.shape)}

    def set_from(self, state, player):
        """Updates observation tensor."""
        obs = self.dict["observation"]
        obs.fill(0)
        
        for row in range(_NUM_ROWS):
            for col in range(_NUM_COLS):
                chess = state.board[row][col]
                idx = 0
                
                # Own pieces (one-hot encoding)
                if chess.country == player and chess.type > 0:
                    # Map type to 0–7 index (skip missing types)
                    type_idx = {1:0, 2:1, 7:2, 8:3, 9:4, 10:5, 12:6}.get(chess.type, -1)
                    if type_idx >= 0:
                        obs[row][col][type_idx] = 1
                
                # Opponent pieces (known or unknown)
                idx = _NUM_CHESS_TYPES
                if chess.country == 1 - player:
                    if chess.type > 0:
                        type_idx = {1:0, 2:1, 7:2, 8:3, 9:4, 10:5, 12:6}.get(chess.type, -1)
                        if type_idx >= 0:
                            obs[row][col][idx + type_idx] = 1
                    else:
                        obs[row][col][idx] = 0.5  # Unknown
                
                # Map type (one-hot)
                idx = 2 * _NUM_CHESS_TYPES
                map_type = state.map[row][col]
                obs[row][col][idx + map_type.value] = 1
                
                # Additional info
                idx = 2 * _NUM_CHESS_TYPES + 4
                obs[row][col][idx] = state.game_length / _GAME_INFO.max_game_length
                obs[row][col][idx + 1] = state.game_length_peace / _NUM_MAX_PEACE_STEP
                obs[row][col][idx + 2] = 1 if state.selecting_piece else 0

    def string_from(self, state, player):
        """Observation of state from the PoV of player, as a string."""
        return str(state)


class Chess:
    def __init__(self, num=-10, country=-1):
        self.num = num
        self.type = ChessType(num if num >= 0 else ChessType.UNKNOWN)
        self.name = _DICT_CHESS_NAME.get(num, "?")
        self.country = country if num != 0 else -1

    def __str__(self):
        if self.type == ChessType.NONE:
            return f"\033[;;m{self.name}\033[0m"
        elif self.country == 0:
            return f"\033[;30;43m{self.name}\033[0m"
        elif self.country == 1:
            return f"\033[;30;42m{self.name}\033[0m"
        else:
            return f"\033[;;m{self.name}\033[0m"

    def __repr__(self):
        return repr(self.num)

    def __eq__(self, other):
        return self.num == other

    def __lt__(self, other):
        return self.num < other

    def __gt__(self, other):
        return self.num > other


def _board_to_string(board, map_grid):
    """Returns a string representation of the board."""
    result = []
    
    # Header with column numbers
    result.append("     " + "    ".join(f"{i}" for i in range(_NUM_COLS)))
    result.append("   +" + "----+" * _NUM_COLS)
    
    for row in range(_NUM_ROWS):
        row_pieces = []
        for col in range(_NUM_COLS):
            chess = board[row][col]
            map_type = map_grid[row][col]
            
            if map_type == MapType.RAILWAY:
                bg = "═"
            elif map_type == MapType.CAMP:
                bg = "△"
            elif map_type == MapType.HQ:
                bg = "◆"
            else:
                bg = " "
            
            if chess.type == ChessType.NONE:
                piece_str = f"{bg}  "
            else:
                piece_str = f"{bg}{chess.name}"
            
            row_pieces.append(piece_str)
        
        row_str = f"{row:2} |" + "|".join(row_pieces) + "|"
        result.append(row_str)
        result.append("   +" + "----+" * _NUM_COLS)
    
    result.append("\n  Map: ═Railway  △Camp  ◆HQ")
    result.append("  Pieces: 司(10) 军(9) 师(8) 旅(7) 工(2) 炸(12) 旗(1)")
    
    return "\n".join(result)


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, JunQi8x3Game)