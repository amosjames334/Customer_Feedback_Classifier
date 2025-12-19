import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional
import pandas as pd

from .config import ModelConfig


class FeedbackDataset(Dataset):
    """Dataset for customer feedback classification."""
    
    def __init__(
        self,
        texts: list[str],
        labels: Optional[list[int]] = None,
        config: Optional[ModelConfig] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.config = config or ModelConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item
    
    @classmethod
    def from_csv(
        cls,
        path: str,
        text_column: str = "text",
        label_column: str = "label",
        config: Optional[ModelConfig] = None,
    ) -> "FeedbackDataset":
        """Load dataset from CSV file."""
        df = pd.read_csv(path)
        texts = df[text_column].tolist()
        labels = df[label_column].tolist() if label_column in df.columns else None
        return cls(texts=texts, labels=labels, config=config)
    
    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "imdb",
        split: str = "train",
        config: Optional[ModelConfig] = None,
        max_samples: Optional[int] = None,
    ) -> "FeedbackDataset":
        """Load dataset from HuggingFace datasets."""
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        texts = dataset["text"]
        
        # Map labels for sentiment (IMDB: 0=neg, 1=pos -> we add neutral)
        if "label" in dataset.features:
            raw_labels = dataset["label"]
            # For IMDB: 0->0 (negative), 1->2 (positive)
            labels = [0 if l == 0 else 2 for l in raw_labels]
        else:
            labels = None
        
        return cls(texts=texts, labels=labels, config=config)

