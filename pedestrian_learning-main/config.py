import dataclasses

@dataclasses.dataclass
class Config:
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 10000
    log_interval: int = 1
    eval_interval: int = 1
    num_pedestrians: int = 100
    eval_fraction: float = 0.1
    pedestrian_hidden_sizes: list[int] = dataclasses.field(default_factory=lambda: [16, 16])
    goal_hidden_sizes: list[int] = dataclasses.field(default_factory=lambda: [16, 16])
    beta1: float = 0.9
    beta2: float = 0.999
    dt: float = 0.1
    experiment_name: str = "experiment"
    dataset_path: str = "dataset.npz"
    init_goal_vel_path: str | None = None
    seed: int = 0

    def __str__(self) -> str:
        lines = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            lines.append(f"{field.name}: {value}")
        return "\n".join(lines)