"""Tests for the feedback classifier model."""

import pytest
import torch
import sys
sys.path.insert(0, ".")

from src.model.config import ModelConfig
from src.model.dataset import FeedbackDataset
from src.model.train import FeedbackClassifier


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.model_name == "distilbert-base-uncased"
        assert config.num_labels == 3
        assert config.max_length == 256
    
    def test_label_mappings(self):
        config = ModelConfig()
        assert config.id2label[0] == "negative"
        assert config.label2id["positive"] == 2


class TestFeedbackDataset:
    def test_dataset_creation(self):
        texts = ["This is great!", "This is terrible."]
        labels = [2, 0]  # positive, negative
        
        dataset = FeedbackDataset(texts=texts, labels=labels)
        
        assert len(dataset) == 2
        
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
    
    def test_dataset_without_labels(self):
        texts = ["Just some text"]
        dataset = FeedbackDataset(texts=texts)
        
        item = dataset[0]
        assert "labels" not in item


class TestFeedbackClassifier:
    def test_model_initialization(self):
        config = ModelConfig()
        model = FeedbackClassifier(config)
        
        assert model.config == config
        assert model.model is not None
    
    def test_forward_pass(self):
        config = ModelConfig()
        model = FeedbackClassifier(config)
        
        # Create dummy input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.tensor([0, 1])
        
        outputs = model(input_ids, attention_mask, labels)
        
        assert hasattr(outputs, "loss")
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, config.num_labels)


class TestAPIIntegration:
    """Integration tests for the API (requires running server)."""
    
    @pytest.mark.skip(reason="Requires running server")
    def test_health_endpoint(self):
        import httpx
        response = httpx.get("http://localhost:8000/health")
        assert response.status_code == 200
    
    @pytest.mark.skip(reason="Requires running server")
    def test_predict_endpoint(self):
        import httpx
        response = httpx.post(
            "http://localhost:8000/predict",
            json={"text": "This product is amazing!"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

