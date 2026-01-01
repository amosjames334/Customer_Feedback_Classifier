import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from typing import Optional
import logging

from .config import ModelConfig
from .dataset import FeedbackDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackClassifier(nn.Module):
    """DistilBERT-based feedback classifier."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=self.config.id2label,
            label2id=self.config.label2id,
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def save(self, path: str):
        """Save model and config."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        
        # Save config
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, config: Optional[ModelConfig] = None):
        """Load model from path."""
        config = config or ModelConfig()
        instance = cls(config)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        return instance


class Trainer:
    """Training loop for the feedback classifier."""
    
    def __init__(
        self,
        model: FeedbackClassifier,
        train_dataset: FeedbackDataset,
        eval_dataset: Optional[FeedbackDataset] = None,
        config: Optional[ModelConfig] = None,
    ):
        self.model = model
        self.config = config or model.config
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        self.eval_loader = None
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        self.history = {"train_loss": [], "eval_loss": [], "eval_accuracy": []}
    
    def train(self) -> dict:
        """Run training loop."""
        logger.info(f"Training on {self.device}")
        logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
        
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            if self.eval_loader:
                eval_metrics = self.evaluate()
                self.history["eval_loss"].append(eval_metrics["loss"])
                self.history["eval_accuracy"].append(eval_metrics["accuracy"])
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Eval Loss: {eval_metrics['loss']:.4f} | "
                    f"Eval Acc: {eval_metrics['accuracy']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f}"
                )
        
        # Save model
        self.model.save(self.config.output_dir)
        
        return self.history
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
        )
        
        for batch in progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> dict:
        """Evaluate model on eval dataset."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        # Get unique labels present in data
        unique_labels = sorted(set(all_labels) | set(all_preds))
        present_names = [self.config.label_names[i] for i in unique_labels if i < len(self.config.label_names)]
        
        return {
            "loss": total_loss / len(self.eval_loader),
            "accuracy": accuracy,
            "f1": f1,
            "report": classification_report(
                all_labels,
                all_preds,
                labels=unique_labels,
                target_names=present_names,
                zero_division=0,
            ),
        }


def train_model(
    dataset_name: str = "imdb",
    max_samples: int = 5000,
    config: Optional[ModelConfig] = None,
) -> FeedbackClassifier:
    """Convenience function to train a model."""
    config = config or ModelConfig()
    
    logger.info(f"Loading dataset: {dataset_name}")
    train_dataset = FeedbackDataset.from_huggingface(
        dataset_name=dataset_name,
        split="train",
        config=config,
        max_samples=max_samples,
    )
    
    eval_dataset = FeedbackDataset.from_huggingface(
        dataset_name=dataset_name,
        split="test",
        config=config,
        max_samples=max_samples // 5,
    )
    
    logger.info("Initializing model")
    model = FeedbackClassifier(config)
    
    logger.info("Starting training")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    
    trainer.train()
    
    # Print final evaluation
    if trainer.eval_loader:
        final_metrics = trainer.evaluate()
        logger.info(f"\nFinal Evaluation:\n{final_metrics['report']}")
    
    return model


if __name__ == "__main__":
    train_model()

