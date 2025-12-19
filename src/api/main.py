import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    use_agent: bool = Field(default=False, description="Use LangGraph agent for complex queries")


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Global model storage
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.id2label = None
    
    def load_model(self, model_path: str = "models/feedback_classifier"):
        """Load the trained model."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            logger.info(f"Model loaded from {model_path} on {self.device}")
            return True
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
            # Load default model for demo
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.model.to(self.device)
            self.model.eval()
            self.id2label = {0: "negative", 1: "positive"}
            logger.info(f"Loaded default SST-2 model on {self.device}")
            return True
    
    def predict(self, text: str) -> PredictionResponse:
        """Make a prediction for a single text."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
        
        probabilities = {
            self.id2label[i]: round(p.item(), 4)
            for i, p in enumerate(probs)
        }
        
        return PredictionResponse(
            text=text,
            label=self.id2label[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            probabilities=probabilities,
        )


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_path = os.getenv("MODEL_PATH", "models/feedback_classifier")
    model_manager.load_model(model_path)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Customer Feedback Classifier API",
    description="ML-powered API for classifying customer feedback sentiment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        device=str(model_manager.device) if model_manager.device else "none",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify a single piece of feedback."""
    if request.use_agent:
        # Import agent here to avoid circular imports
        from src.agent.langgraph_agent import run_agent
        result = await run_agent(request.text)
        return result
    
    return model_manager.predict(request.text)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Classify multiple pieces of feedback."""
    predictions = [model_manager.predict(text) for text in request.texts]
    return BatchPredictionResponse(predictions=predictions)


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "device": str(model_manager.device),
        "labels": model_manager.id2label,
        "model_type": type(model_manager.model).__name__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

