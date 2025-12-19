from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the feedback classifier model."""
    
    # Model settings
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 3  # positive, negative, neutral
    max_length: int = 256
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Paths
    output_dir: str = "models/feedback_classifier"
    data_dir: str = "data"
    
    # Labels
    label_names: tuple = ("negative", "neutral", "positive")
    
    # Device
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def id2label(self) -> dict:
        return {i: label for i, label in enumerate(self.label_names)}
    
    @property
    def label2id(self) -> dict:
        return {label: i for i, label in enumerate(self.label_names)}

