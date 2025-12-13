from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import yaml
import os


@dataclass
class GameConfig:
    """Game environment configuration."""
    mode: str = "junqi_8x3"  # or "junqi_standard"
    board_variant: str = "small"  # or "standard" (derived from mode)
    
    def __post_init__(self):
        """Auto-set board_variant based on game mode if not explicitly set."""
        if self.board_variant == "small" and self.mode == "junqi_standard":
            self.board_variant = "standard"
        elif self.board_variant == "standard" and self.mode == "junqi_8x3":
            self.board_variant = "small"


@dataclass
class TrainingConfig:
    """General training hyperparameters."""
    num_episodes: int = 5000
    num_iterations: int = 5000  # For PPO training
    eval_every: int = 1000
    eval_episodes: int = 100
    save_dir: str = "models"
    device: str = "cpu"
    seed: Optional[int] = None


@dataclass
class DRQNConfig:
    """DRQN-specific hyperparameters."""
    # Network architecture
    network_type: str = "flat"  # "flat" or "spatial"
    hidden_size: int = 256
    
    # Training hyperparameters
    lr: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.20
    epsilon_decay: float = 0.9995
    
    # Experience replay
    batch_size: int = 64
    replay_buffer_size: int = 100000
    use_stratified_buffer: bool = True
    num_segments: int = 4
    
    # Target network and opponent
    target_update_freq: int = 1000
    opponent_update_freq: int = 500


@dataclass  
class TransformerConfig:
    """Transformer PPO hyperparameters."""
    # Model architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # PPO hyperparameters
    lr_start: float = 1e-4
    lr_end: float = 5e-6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.001
    vf_coef: float = 0.5
    
    # Training loop
    num_steps: int = 512
    minibatch_size: int = 32
    update_epochs: int = 4
    max_grad_norm: float = 0.5

@dataclass
class RPPOConfig:
    """Recurrent PPO (RPPO) hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    k_epochs: int = 4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Network
    hidden_size: int = 256

    # Training
    batch_size: int = 64



@dataclass
class WandbConfig:
    """WandB configuration."""
    enabled: bool = True
    project: str = "junqi-drl"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class Config:
    def __init__(
        self,
        game: Optional[GameConfig] = None,
        training: Optional[TrainingConfig] = None,
        wandb: Optional[WandbConfig] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        agent_type: str = "drqn"
    ):
        """
        Initialize configuration.
        
        Args:
            game: Game configuration
            training: Training configuration
            wandb: Wandb configuration
            agent_config: Agent-specific configuration dict
            agent_type: Type of agent ("drqn" or "transformer")
        """
        self.game = game or GameConfig()
        self.training = training or TrainingConfig()
        self.wandb = wandb or WandbConfig()
        self.agent_type = agent_type
        
        # Initialize agent-specific config based on type
        if agent_config is not None:
            self.agent_config = agent_config
        else:
            if agent_type == "drqn":
                self.agent_config = DRQNConfig()
            elif agent_type == "transformer":
                self.agent_config = TransformerConfig()
            elif agent_type == "rppo":
                self.agent_config = RPPOConfig()
            else:
                self.agent_config = {}
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            Config object
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse game config
        game_data = data.get('game', {})
        game_config = GameConfig(**game_data)
        
        # Parse training config
        training_data = data.get('training', {})
        training_config = TrainingConfig(**training_data)
        
        # Parse wandb config
        wandb_data = data.get('wandb', {})
        wandb_config = WandbConfig(**wandb_data)
        
        # Parse agent config based on type
        agent_data = data.get('agent', {})
        agent_type = agent_data.get('type', 'drqn')
        
        if agent_type == 'drqn':
            agent_config = DRQNConfig(**{k: v for k, v in agent_data.items() if k != 'type'})
        elif agent_type == 'transformer':
            agent_config = TransformerConfig(**{k: v for k, v in agent_data.items() if k != 'type'})
        elif agent_type == 'rppo':
            agent_config = RPPOConfig(**{k: v for k, v in agent_data.items() if k != 'type'})
        else:
            agent_config = agent_data
        
        return cls(
            game=game_config,
            training=training_config,
            wandb=wandb_config,
            agent_config=agent_config,
            agent_type=agent_type
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for logging.
        
        Returns:
            Dictionary representation of config
        """
        result = {
            'game': asdict(self.game),
            'training': asdict(self.training),
            'wandb': asdict(self.wandb),
            'agent_type': self.agent_type
        }
        
        if isinstance(self.agent_config, (DRQNConfig, TransformerConfig, RPPOConfig)):
            result['agent'] = asdict(self.agent_config)
        else:
            result['agent'] = self.agent_config
        
        return result
    
    def save(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config(agent_type={self.agent_type}, game={self.game.mode})"
