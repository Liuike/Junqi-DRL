"""Observation processing utilities for Junqi game states."""

import numpy as np
import torch
from typing import Dict, Tuple


def get_board_config(game_mode: str) -> Dict[str, int]:
    """
    Get board configuration for a game mode.
    
    Args:
        game_mode: "junqi_8x3" or "junqi_standard"
        
    Returns:
        Dictionary with 'height', 'width', 'channels'
    """
    configs = {
        "junqi_8x3": {"height": 8, "width": 3, "channels": 23},
        "junqi_standard": {"height": 12, "width": 5, "channels": 31},
    }
    
    if game_mode not in configs:
        raise ValueError(f"Unknown game mode: {game_mode}. Expected 'junqi_8x3' or 'junqi_standard'")
    
    return configs[game_mode]


def process_observation(
    state, 
    player_id: int, 
    board_config: Dict[str, int],
    device: str = "cpu",
    format: str = "spatial",
    add_batch_dim: bool = True,
    add_seq_dim: bool = False
) -> torch.Tensor:
    """
    Process game state into tensor observation.
    
    This function handles the conversion from OpenSpiel's flat observation format
    to either spatial (H, W, C) or flat (H*W*C,) PyTorch tensors.
    
    Args:
        state: OpenSpiel game state
        player_id: Player perspective (0 or 1)
        board_config: Dict with 'height', 'width', 'channels'
        device: PyTorch device ("cpu" or "cuda")
        format: "spatial" preserves (H,W,C), "flat" flattens to (H*W*C,)
        add_batch_dim: Add batch dimension
        add_seq_dim: Add sequence dimension for recurrent models
        
    Returns:
        Processed observation tensor with shape depending on options:
        - format="spatial", add_batch_dim=True: (1, H, W, C)
        - format="spatial", add_seq_dim=True: (1, 1, H, W, C)
        - format="flat", add_batch_dim=True, add_seq_dim=True: (1, 1, H*W*C)
    
    Example:
        >>> config = get_board_config("junqi_8x3")
        >>> obs = process_observation(state, 0, config, format="spatial")
        >>> obs.shape
        torch.Size([1, 8, 3, 23])
    """
    # Get flat observation from OpenSpiel
    obs_flat = state.observation_tensor(player_id)
    obs_array = np.array(obs_flat, dtype=np.float32)
    
    # Reshape to spatial structure
    obs_3d = obs_array.reshape(
        board_config['height'],
        board_config['width'],
        board_config['channels']
    )
    
    if format == "flat":
        # Flatten for traditional MLPs
        obs_tensor = torch.tensor(obs_array, device=device, dtype=torch.float32)
        if add_batch_dim and add_seq_dim:
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H*W*C)
        elif add_batch_dim:
            obs_tensor = obs_tensor.unsqueeze(0)  # (1, H*W*C)
        elif add_seq_dim:
            obs_tensor = obs_tensor.unsqueeze(0)  # (1, H*W*C)
    else:
        # Keep spatial structure for CNNs/Transformers
        obs_tensor = torch.tensor(obs_3d, device=device, dtype=torch.float32)
        if add_batch_dim and add_seq_dim:
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, C)
        elif add_batch_dim:
            obs_tensor = obs_tensor.unsqueeze(0)  # (1, H, W, C)
        elif add_seq_dim:
            obs_tensor = obs_tensor.unsqueeze(0)  # (1, H, W, C)
    
    return obs_tensor


def observation_shape(
    game_mode: str,
    format: str = "spatial"
) -> Tuple[int, ...]:
    """
    Get expected observation shape for a game mode.
    
    Args:
        game_mode: "junqi_8x3" or "junqi_standard"
        format: "spatial" or "flat"
        
    Returns:
        Tuple representing shape (without batch/sequence dimensions)
    
    Example:
        >>> observation_shape("junqi_8x3", "spatial")
        (8, 3, 23)
        >>> observation_shape("junqi_8x3", "flat")
        (552,)
    """
    config = get_board_config(game_mode)
    
    if format == "spatial":
        return (config['height'], config['width'], config['channels'])
    else:
        return (config['height'] * config['width'] * config['channels'],)


def flatten_observation(obs_spatial: torch.Tensor) -> torch.Tensor:
    """
    Flatten spatial observation to 1D.
    
    Args:
        obs_spatial: Spatial observation of shape (..., H, W, C)
        
    Returns:
        Flattened observation of shape (..., H*W*C)
    """
    batch_dims = obs_spatial.shape[:-3]
    h, w, c = obs_spatial.shape[-3:]
    return obs_spatial.reshape(*batch_dims, h * w * c)


def unflatten_observation(
    obs_flat: torch.Tensor,
    board_config: Dict[str, int]
) -> torch.Tensor:
    """
    Reshape flat observation back to spatial format.
    
    Args:
        obs_flat: Flat observation of shape (..., H*W*C)
        board_config: Dict with 'height', 'width', 'channels'
        
    Returns:
        Spatial observation of shape (..., H, W, C)
    """
    batch_dims = obs_flat.shape[:-1]
    return obs_flat.reshape(
        *batch_dims,
        board_config['height'],
        board_config['width'],
        board_config['channels']
    )
