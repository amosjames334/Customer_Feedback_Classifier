#!/usr/bin/env python3
"""Training script for the feedback classifier."""

import argparse
import sys
sys.path.insert(0, ".")

from src.model.config import ModelConfig
from src.model.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train the feedback classifier")
    parser.add_argument("--dataset", default="imdb", help="Dataset name (HuggingFace)")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="models/feedback_classifier", help="Output directory")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base model")
    
    args = parser.parse_args()
    
    config = ModelConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    
    model = train_model(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        config=config,
    )
    
    print(f"\nModel saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

