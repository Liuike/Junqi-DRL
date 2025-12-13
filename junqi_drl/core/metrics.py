from typing import Dict, Any, Optional
from collections import defaultdict
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MetricsLogger:
    def __init__(
        self,
        use_wandb: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metrics logger.
        
        Args:
            use_wandb: Whether to use Weights & Biases logging
            wandb_config: WandB configuration dict (project, entity, run_name, etc.)
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.metrics_history = defaultdict(list)
        self.wandb_initialized = False
        
        if self.use_wandb:
            self._init_wandb(wandb_config or {})
        elif use_wandb and not WANDB_AVAILABLE:
            print("Warning: WandB requested but not available. Logging to console only.")
    
    def _init_wandb(self, config: Dict[str, Any]):
        """
        Initialize Weights & Biases.
        
        Args:
            config: WandB configuration
        """
        try:
            wandb.init(
                project=config.get('project', 'junqi-drl'),
                entity=config.get('entity'),
                name=config.get('name'),
                tags=config.get('tags', []),
                config=config.get('config', {})
            )
            self.wandb_initialized = True
            print(f"WandB initialized: {wandb.run.name}")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB: {e}")
            self.use_wandb = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to console and WandB.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/episode number for x-axis
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.metrics_history[key].append(value)
        
        if self.use_wandb and self.wandb_initialized:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to WandB: {e}")
    
    def log_summary(self, summary: Dict[str, Any]):
        """
        Log final summary statistics to WandB.
        
        Args:
            summary: Dictionary of summary metrics
        """
        if self.use_wandb and self.wandb_initialized:
            try:
                wandb.summary.update(summary)
            except Exception as e:
                print(f"Warning: Failed to log summary to WandB: {e}")
    
    def get_history(self, key: str) -> list:
        """
        Retrieve metric history.
        
        Args:
            key: Metric name
            
        Returns:
            List of metric values over time
        """
        return self.metrics_history.get(key, [])
    
    def get_recent_mean(self, key: str, window: int = 100) -> Optional[float]:
        """
        Get mean of recent metric values.
        
        Args:
            key: Metric name
            window: Number of recent values to average
            
        Returns:
            Mean value or None if insufficient data
        """
        history = self.metrics_history.get(key, [])
        if not history:
            return None
        recent = history[-window:]
        return np.mean(recent)
    
    def finish(self):
        """Clean up logging resources."""
        if self.use_wandb and self.wandb_initialized:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish WandB: {e}")


def print_metrics_summary(
    metrics: Dict[str, Any],
    title: str = "Metrics",
    width: int = 60
):
    """
    Pretty print metrics summary.
    
    Args:
        metrics: Dictionary of metric names to values
        title: Title for the summary
        width: Width of the separator line
    """
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 0 < abs(value) < 0.01 or abs(value) > 1000:
                print(f"{key:30s}: {value:.4e}")
            else:
                print(f"{key:30s}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{key:30s}: {value}")
        else:
            print(f"{key:30s}: {value}")
    
    print(f"{'='*width}\n")
