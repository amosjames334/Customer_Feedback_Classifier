#!/usr/bin/env python3
"""Deployment script for SageMaker."""

import argparse
import sys
sys.path.insert(0, ".")

from src.sagemaker.deploy import deploy_model


def main():
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker")
    parser.add_argument("--model-dir", default="models/feedback_classifier", help="Model directory")
    parser.add_argument("--endpoint-name", default="feedback-classifier", help="Endpoint name")
    parser.add_argument("--instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--serverless", action="store_true", help="Use serverless inference")
    
    args = parser.parse_args()
    
    endpoint = deploy_model(
        model_dir=args.model_dir,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        serverless=args.serverless,
    )
    
    print(f"\nEndpoint deployed: {endpoint}")


if __name__ == "__main__":
    main()

