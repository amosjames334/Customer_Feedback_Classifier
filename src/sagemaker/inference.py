import json
import boto3
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerPredictor:
    """Client for making predictions against SageMaker endpoint."""
    
    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.client = boto3.client(
            "sagemaker-runtime",
            region_name=region,
        )
    
    def predict(self, text: str) -> dict:
        """Make a prediction for a single text."""
        payload = {"inputs": text}
        
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        
        result = json.loads(response["Body"].read().decode())
        return self._parse_result(result, text)
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Make predictions for multiple texts."""
        payload = {"inputs": texts}
        
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        
        results = json.loads(response["Body"].read().decode())
        return [
            self._parse_result(result, text)
            for result, text in zip(results, texts)
        ]
    
    def _parse_result(self, result: list[dict], text: str) -> dict:
        """Parse HuggingFace inference result."""
        if isinstance(result, list) and len(result) > 0:
            # Result is list of label/score pairs
            best = max(result, key=lambda x: x.get("score", 0))
            probabilities = {
                item["label"]: round(item["score"], 4)
                for item in result
            }
            return {
                "text": text,
                "label": best["label"],
                "confidence": round(best["score"], 4),
                "probabilities": probabilities,
            }
        
        return {"text": text, "label": "unknown", "confidence": 0.0}
    
    async def predict_async(self, text: str) -> dict:
        """Async prediction wrapper."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, text)


class SageMakerBatchTransform:
    """Run batch inference using SageMaker Batch Transform."""
    
    def __init__(
        self,
        model_name: str,
        region: str = "us-east-1",
        role_arn: Optional[str] = None,
    ):
        self.model_name = model_name
        self.region = region
        self.role_arn = role_arn
        
        import sagemaker
        self.session = sagemaker.Session()
        self.bucket = self.session.default_bucket()
    
    def run_batch_transform(
        self,
        input_s3_uri: str,
        output_s3_uri: str,
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
    ) -> str:
        """Run batch transform job."""
        import sagemaker
        from sagemaker.transformer import Transformer
        
        transformer = Transformer(
            model_name=self.model_name,
            instance_count=instance_count,
            instance_type=instance_type,
            output_path=output_s3_uri,
            sagemaker_session=self.session,
        )
        
        transformer.transform(
            data=input_s3_uri,
            content_type="application/json",
            split_type="Line",
        )
        
        transformer.wait()
        logger.info(f"Batch transform complete. Output: {output_s3_uri}")
        
        return output_s3_uri

