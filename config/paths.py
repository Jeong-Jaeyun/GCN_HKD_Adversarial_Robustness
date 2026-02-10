"""Path configuration for dataset and model management."""

from pathlib import Path
from typing import Optional


class PathConfig:
    """Manages all system paths in a centralized manner."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root = Path(root_dir) if root_dir else Path(__file__).parent.parent
        
        # Data directories
        self.data_dir = self.root / "data" / "raw"
        self.processed_data_dir = self.root / "data" / "processed"
        self.graphs_dir = self.root / "data" / "graphs"
        
        # Model directories
        self.models_dir = self.root / "checkpoints"
        self.teacher_model_dir = self.models_dir / "teacher"
        self.student_model_dir = self.models_dir / "student"
        
        # Log and output directories
        self.logs_dir = self.root / "logs"
        self.results_dir = self.root / "results"
        self.visualizations_dir = self.results_dir / "visualizations"
        
        # Dataset specific paths
        self.sard_dir = self.data_dir / "SARD"
        self.bigvul_dir = self.data_dir / "BigVul"
        self.dsieve_dir = self.data_dir / "D-Sieve"
        
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create all necessary directories."""
        for path_attr in dir(self):
            if not path_attr.startswith('_') and 'dir' in path_attr:
                path = getattr(self, path_attr)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, model_name: str, epoch: int) -> Path:
        """Get checkpoint path for a specific model and epoch."""
        return self.models_dir / f"{model_name}_epoch{epoch}.pt"
    
    def get_log_path(self, experiment_name: str) -> Path:
        """Get log path for an experiment."""
        return self.logs_dir / f"{experiment_name}.log"


# Default instance
default_paths = PathConfig()
