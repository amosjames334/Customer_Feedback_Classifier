# Customer Feedback Classifier

Production-grade ML system for customer feedback sentiment classification using PyTorch, AWS SageMaker, and LangGraph.

## Features

- **Fine-tuned DistilBERT** for sentiment classification (PyTorch + HuggingFace)
- **FastAPI Backend** with async support and batch predictions
- **LangGraph Agent** for intelligent query routing and complex analysis
- **AWS SageMaker Deployment** with auto-scaling and serverless options
- **CloudWatch Monitoring** with custom dashboards and alerts

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│                   /predict, /health                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│   LangGraph      │    │        AWS SageMaker                 │
│   Agent Flow     │    │   ┌────────────────────────────┐     │
│                  │    │   │  Fine-tuned DistilBERT     │     │
│  • Route queries │    │   │  (PyTorch)                 │     │
│  • Multi-step    │    │   └────────────────────────────┘     │
│    reasoning     │    │                                      │
└──────────────────┘    └─────────────────┬────────────────────┘
                                          │
                                          ▼
                        ┌─────────────────────────────────────┐
                        │         MLOps Layer                 │
                        │  • CloudWatch Monitoring            │
                        │  • Model versioning (S3)            │
                        │  • Performance logging              │
                        └─────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py --max-samples 5000

# Run API
uvicorn src.api.main:app --reload

# Test
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Excellent product, highly recommend!"}'
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model Training | PyTorch + HuggingFace Transformers |
| Model Type | DistilBERT |
| Deployment | AWS SageMaker |
| Orchestration | LangGraph |
| Monitoring | CloudWatch |
| API Layer | FastAPI |

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This product exceeded my expectations!"}'
```

Response:
```json
{
    "text": "This product exceeded my expectations!",
    "label": "positive",
    "confidence": 0.9821,
    "probabilities": {
        "negative": 0.0089,
        "neutral": 0.009,
        "positive": 0.9821
    }
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Great service!", "Terrible experience", "It was okay"]}'
```

### With LangGraph Agent

```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Why are customers unhappy with shipping?", "use_agent": true}'
```

## Documentation

- [Setup Guide](setup.md) - Detailed installation and configuration
- [Project Spec](project.md) - Original project specification

## License

MIT
