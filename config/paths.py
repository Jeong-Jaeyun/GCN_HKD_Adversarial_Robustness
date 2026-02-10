
from pathlib import Path
from typing import Optional


class PathConfig:

    def __init__(self, root_dir: Optional[Path] = None):
        self.root = Path(root_dir) if root_dir else Path(__file__).parent.parent


        self.data_dir = self.root / "data" / "raw"
        self.processed_data_dir = self.root / "data" / "processed"
        self.graphs_dir = self.root / "data" / "graphs"


        self.models_dir = self.root / "checkpoints"
        self.teacher_model_dir = self.models_dir / "teacher"
        self.student_model_dir = self.models_dir / "student"


        self.logs_dir = self.root / "logs"
        self.results_dir = self.root / "results"
        self.visualizations_dir = self.results_dir / "visualizations"


        self.sard_dir = self.data_dir / "SARD"
        self.bigvul_dir = self.data_dir / "BigVul"
        self.dsieve_dir = self.data_dir / "D-Sieve"

        self._create_directories()

    def _create_directories(self) -> None:
        for path_attr in dir(self):
            if not path_attr.startswith('_') and 'dir' in path_attr:
                path = getattr(self, path_attr)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, model_name: str, epoch: int) -> Path:
        return self.models_dir / f"{model_name}_epoch{epoch}.pt"

    def get_log_path(self, experiment_name: str) -> Path:
        return self.logs_dir / f"{experiment_name}.log"


default_paths = PathConfig()
