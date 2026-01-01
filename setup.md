# Setup Guide

## Prerequisites

- Python 3.10+
- AWS Account (for SageMaker deployment)
- OpenAI API key (optional, for LangGraph agent)

## 1. Environment Setup

```bash
# Clone and navigate
cd Customer_Feedback_Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Configuration

Create a `.env` file:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole

# OpenAI (for LangGraph agent)
OPENAI_API_KEY=your_openai_key

# Model Configuration
MODEL_PATH=models/feedback_classifier
```

## 3. Train the Model

```bash
# Quick training (5000 samples, ~10 min on GPU)
python scripts/train.py

# Full training options
python scripts/train.py \
    --dataset imdb \
    --max-samples 10000 \
    --epochs 3 \
    --batch-size 16 \
    --output-dir models/feedback_classifier
```

## 4. Run the API Locally

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Test the API
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This product is amazing!"}'
```

## 5. Deploy to SageMaker

### Create IAM Role

Create a SageMaker execution role with these policies:
- AmazonSageMakerFullAccess
- AmazonS3FullAccess

### Deploy

```bash
# Standard deployment
python scripts/deploy.py \
    --model-dir models/feedback_classifier \
    --endpoint-name feedback-classifier \
    --instance-type ml.m5.large

# Serverless deployment (cost-effective)
python scripts/deploy.py --serverless
```

### Test SageMaker Endpoint

```python
from src.sagemaker.inference import SageMakerPredictor

predictor = SageMakerPredictor(endpoint_name="your-endpoint-name")
result = predictor.predict("Great customer service!")
print(result)
```

## 6. Set Up Monitoring

```python
from src.monitoring.cloudwatch import CloudWatchMonitor

monitor = CloudWatchMonitor(endpoint_name="your-endpoint-name")

# Create dashboard
monitor.create_dashboard("FeedbackClassifier")

# Create latency alarm
monitor.create_alarm(
    alarm_name="HighLatency",
    metric_name="ModelLatency",
    threshold=1000,  # ms
)
```

## 7. Use LangGraph Agent

```python
from src.agent.langgraph_agent import FeedbackAgent

agent = FeedbackAgent()
result = await agent.run("Why is this product getting negative reviews?")
print(result)
```

## Docker Deployment

```bash
# Build
docker build -t feedback-classifier .

# Run
docker run -p 8000:8000 \
    -e MODEL_PATH=models/feedback_classifier \
    feedback-classifier
```

## Project Structure

```
Customer_Feedback_Classifier/
├── src/
│   ├── model/          # PyTorch model training
│   ├── api/            # FastAPI backend
│   ├── agent/          # LangGraph agent
│   ├── sagemaker/      # AWS deployment
│   └── monitoring/     # CloudWatch monitoring
├── scripts/            # Training & deployment scripts
├── tests/              # Unit tests
├── models/             # Saved models (created after training)
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model information |

## Troubleshooting

**Model not found error:**
- Run training first: `python scripts/train.py`
- Or set `MODEL_PATH` to a pretrained model


**SageMaker deployment fails:**
- Verify IAM role has correct permissions
- Check S3 bucket exists and is accessible
- Ensure model artifacts are valid

