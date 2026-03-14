from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class FixedAllocationConfig:
    subtitle: int = 2
    frame: int = 2
    segment: int = 2


@dataclass(slots=True)
class BudgetConfig:
    max_items: int = 6
    subtitle_cost: float = 1.0
    frame_cost: float = 1.0
    segment_cost: float = 1.5


@dataclass(slots=True)
class RetrievalConfig:
    name: str = "bm25"
    candidate_pool_size_per_modality: int = 16
    fixed_allocation: FixedAllocationConfig = field(default_factory=FixedAllocationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)


@dataclass(slots=True)
class AnswererConfig:
    name: str = "trainable_linear"
    confidence_threshold: float = 0.65
    text_feature_dim: int = 4096
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.2
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 13


@dataclass(slots=True)
class PolicyConfig:
    name: str = "trainable_linear_policy"
    max_items: int = 6
    min_items_before_stop: int = 1
    text_feature_dim: int = 4096
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.2
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 13


@dataclass(slots=True)
class ProjectConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    answerer: AnswererConfig = field(default_factory=AnswererConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def to_dict(self) -> dict:
        return asdict(self)
