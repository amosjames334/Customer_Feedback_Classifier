from __future__ import annotations

import os
import json
import tarfile
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from datetime import datetime
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """Deploy models to AWS SageMaker."""
    
    def __init__(
        self,
        role_arn: Optional[str] = None,
        region: str = "us-east-1",
        bucket_name: Optional[str] = None,
    ):
        self.region = region
        self.role_arn = role_arn or os.getenv("SAGEMAKER_ROLE_ARN")
        
        self.session = boto3.Session(region_name=region)
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        
        self.bucket_name = bucket_name or self.sagemaker_session.default_bucket()
        self.s3_client = self.session.client("s3")
    
    def package_model(
        self,
        model_dir: str,
        output_path: str = "model.tar.gz",
    ) -> str:
        """Package model artifacts for SageMaker."""
        logger.info(f"Packaging model from {model_dir}")
        
        with tarfile.open(output_path, "w:gz") as tar:
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                tar.add(file_path, arcname=file)
        
        logger.info(f"Model packaged to {output_path}")
        return output_path
    
    def upload_to_s3(
        self,
        local_path: str,
        s3_prefix: str = "models/feedback-classifier",
    ) -> str:
        """Upload model artifacts to S3."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_key = f"{s3_prefix}/{timestamp}/model.tar.gz"
        
        logger.info(f"Uploading to s3://{self.bucket_name}/{s3_key}")
        self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
        
        s3_uri = f"s3://{self.bucket_name}/{s3_key}"
        logger.info(f"Uploaded to {s3_uri}")
        return s3_uri
    
    def deploy_endpoint(
        self,
        model_s3_uri: str,
        endpoint_name: str = "feedback-classifier",
        instance_type: str = "ml.m5.large",
        instance_count: int = 1,
        transformers_version: str = "4.26",
        pytorch_version: str = "1.13",
        py_version: str = "py39",
    ) -> str:
        """Deploy model to SageMaker endpoint."""
        if not self.role_arn:
            raise ValueError("SAGEMAKER_ROLE_ARN not set")
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        endpoint_name = f"{endpoint_name}-{timestamp}"
        
        logger.info(f"Deploying to endpoint: {endpoint_name}")
        
        huggingface_model = HuggingFaceModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            transformers_version=transformers_version,
            pytorch_version=pytorch_version,
            py_version=py_version,
            sagemaker_session=self.sagemaker_session,
        )
        
        predictor = huggingface_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
        
        logger.info(f"Endpoint deployed: {endpoint_name}")
        return endpoint_name
    
    def deploy_serverless(
        self,
        model_s3_uri: str,
        endpoint_name: str = "feedback-classifier-serverless",
        memory_size: int = 4096,
        max_concurrency: int = 10,
    ) -> str:
        """Deploy model to SageMaker Serverless Inference."""
        if not self.role_arn:
            raise ValueError("SAGEMAKER_ROLE_ARN not set")
        
        from sagemaker.serverless import ServerlessInferenceConfig
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        endpoint_name = f"{endpoint_name}-{timestamp}"
        
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=memory_size,
            max_concurrency=max_concurrency,
        )
        
        huggingface_model = HuggingFaceModel(
            model_data=model_s3_uri,
            role=self.role_arn,
            transformers_version="4.26",
            pytorch_version="1.13",
            py_version="py39",
            sagemaker_session=self.sagemaker_session,
        )
        
        predictor = huggingface_model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name,
        )
        
        logger.info(f"Serverless endpoint deployed: {endpoint_name}")
        return endpoint_name
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete a SageMaker endpoint."""
        sm_client = self.session.client("sagemaker")
        
        try:
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Deleted endpoint: {endpoint_name}")
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            raise
    
    def list_endpoints(self) -> List[Dict]:
        """List all SageMaker endpoints."""
        sm_client = self.session.client("sagemaker")
        response = sm_client.list_endpoints()
        return response.get("Endpoints", [])


def deploy_model(
    model_dir: str = "models/feedback_classifier",
    endpoint_name: str = "feedback-classifier",
    instance_type: str = "ml.m5.large",
    serverless: bool = False,
) -> str:
    """Convenience function to deploy a model."""
    deployer = SageMakerDeployer()
    
    # Package model
    tar_path = deployer.package_model(model_dir)
    
    # Upload to S3
    s3_uri = deployer.upload_to_s3(tar_path)
    
    # Deploy
    if serverless:
        endpoint = deployer.deploy_serverless(s3_uri, endpoint_name)
    else:
        endpoint = deployer.deploy_endpoint(s3_uri, endpoint_name, instance_type)
    
    # Cleanup local tar
    os.remove(tar_path)
    
    return endpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/feedback_classifier")
    parser.add_argument("--endpoint-name", default="feedback-classifier")
    parser.add_argument("--serverless", action="store_true")
    args = parser.parse_args()
    
    endpoint = deploy_model(
        model_dir=args.model_dir,
        endpoint_name=args.endpoint_name,
        serverless=args.serverless,
    )
    print(f"Deployed endpoint: {endpoint}")

