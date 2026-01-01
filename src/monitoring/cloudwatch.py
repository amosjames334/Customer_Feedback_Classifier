from __future__ import annotations

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import boto3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsLogger:
    """Log custom metrics to CloudWatch."""
    
    def __init__(
        self,
        namespace: str = "FeedbackClassifier",
        region: str = "us-east-1",
    ):
        self.namespace = namespace
        self.client = boto3.client("cloudwatch", region_name=region)
        self._batch = []
        self._batch_size = 20  # CloudWatch limit
    
    def log_metric(
        self,
        name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[Dict[str, str]] = None,
    ):
        """Log a single metric."""
        metric = {
            "MetricName": name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.utcnow(),
        }
        
        if dimensions:
            metric["Dimensions"] = [
                {"Name": k, "Value": v} for k, v in dimensions.items()
            ]
        
        self._batch.append(metric)
        
        if len(self._batch) >= self._batch_size:
            self.flush()
    
    def log_latency(
        self,
        latency_ms: float,
        endpoint: str = "default",
    ):
        """Log inference latency."""
        self.log_metric(
            name="InferenceLatency",
            value=latency_ms,
            unit="Milliseconds",
            dimensions={"Endpoint": endpoint},
        )
    
    def log_prediction(
        self,
        label: str,
        confidence: float,
        endpoint: str = "default",
    ):
        """Log prediction metrics."""
        self.log_metric(
            name="PredictionConfidence",
            value=confidence,
            unit="None",
            dimensions={"Endpoint": endpoint, "Label": label},
        )
        
        self.log_metric(
            name="PredictionCount",
            value=1,
            unit="Count",
            dimensions={"Endpoint": endpoint, "Label": label},
        )
    
    def log_error(self, error_type: str, endpoint: str = "default"):
        """Log error occurrence."""
        self.log_metric(
            name="ErrorCount",
            value=1,
            unit="Count",
            dimensions={"Endpoint": endpoint, "ErrorType": error_type},
        )
    
    def flush(self):
        """Flush batched metrics to CloudWatch."""
        if not self._batch:
            return
        
        try:
            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=self._batch,
            )
            logger.debug(f"Flushed {len(self._batch)} metrics")
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
        finally:
            self._batch = []


class CloudWatchMonitor:
    """Monitor SageMaker endpoints and model performance."""
    
    def __init__(
        self,
        endpoint_name: str,
        region: str = "us-east-1",
        namespace: str = "FeedbackClassifier",
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.namespace = namespace
        
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.sagemaker = boto3.client("sagemaker", region_name=region)
        self.logs = boto3.client("logs", region_name=region)
    
    def get_endpoint_metrics(
        self,
        hours: int = 24,
        period: int = 300,  # 5 minutes
    ) -> Dict:
        """Get SageMaker endpoint metrics."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = {}
        metric_names = [
            "Invocations",
            "InvocationsPerInstance",
            "ModelLatency",
            "OverheadLatency",
            "Invocation4XXErrors",
            "Invocation5XXErrors",
        ]
        
        for metric_name in metric_names:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName=metric_name,
                    Dimensions=[
                        {"Name": "EndpointName", "Value": self.endpoint_name},
                        {"Name": "VariantName", "Value": "AllTraffic"},
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=period,
                    Statistics=["Average", "Sum", "Maximum"],
                )
                
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    metrics[metric_name] = {
                        "average": sum(d.get("Average", 0) for d in datapoints) / len(datapoints),
                        "max": max(d.get("Maximum", 0) for d in datapoints),
                        "total": sum(d.get("Sum", 0) for d in datapoints),
                    }
            except Exception as e:
                logger.warning(f"Could not get metric {metric_name}: {e}")
        
        return metrics
    
    def get_custom_metrics(
        self,
        metric_name: str,
        hours: int = 24,
        period: int = 300,
    ) -> List[Dict]:
        """Get custom application metrics."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        response = self.cloudwatch.get_metric_statistics(
            Namespace=self.namespace,
            MetricName=metric_name,
            Dimensions=[
                {"Name": "Endpoint", "Value": self.endpoint_name},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=["Average", "Sum", "Maximum", "Minimum"],
        )
        
        return response.get("Datapoints", [])
    
    def create_dashboard(self, dashboard_name: str = "FeedbackClassifier"):
        """Create a CloudWatch dashboard for monitoring."""
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0,
                    "width": 12, "height": 6,
                    "properties": {
                        "title": "Invocations",
                        "metrics": [
                            ["AWS/SageMaker", "Invocations",
                             "EndpointName", self.endpoint_name,
                             "VariantName", "AllTraffic"],
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region,
                    },
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0,
                    "width": 12, "height": 6,
                    "properties": {
                        "title": "Model Latency",
                        "metrics": [
                            ["AWS/SageMaker", "ModelLatency",
                             "EndpointName", self.endpoint_name,
                             "VariantName", "AllTraffic"],
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                    },
                },
                {
                    "type": "metric",
                    "x": 0, "y": 6,
                    "width": 12, "height": 6,
                    "properties": {
                        "title": "Errors",
                        "metrics": [
                            ["AWS/SageMaker", "Invocation4XXErrors",
                             "EndpointName", self.endpoint_name,
                             "VariantName", "AllTraffic"],
                            ["AWS/SageMaker", "Invocation5XXErrors",
                             "EndpointName", self.endpoint_name,
                             "VariantName", "AllTraffic"],
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.region,
                    },
                },
                {
                    "type": "metric",
                    "x": 12, "y": 6,
                    "width": 12, "height": 6,
                    "properties": {
                        "title": "Prediction Confidence",
                        "metrics": [
                            [self.namespace, "PredictionConfidence",
                             "Endpoint", self.endpoint_name],
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                    },
                },
            ],
        }
        
        self.cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body),
        )
        
        logger.info(f"Dashboard created: {dashboard_name}")
    
    def create_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "GreaterThanThreshold",
        evaluation_periods: int = 2,
        sns_topic_arn: Optional[str] = None,
    ):
        """Create a CloudWatch alarm."""
        alarm_config = {
            "AlarmName": alarm_name,
            "MetricName": metric_name,
            "Namespace": "AWS/SageMaker",
            "Dimensions": [
                {"Name": "EndpointName", "Value": self.endpoint_name},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            "Statistic": "Average",
            "Period": 300,
            "EvaluationPeriods": evaluation_periods,
            "Threshold": threshold,
            "ComparisonOperator": comparison,
        }
        
        if sns_topic_arn:
            alarm_config["AlarmActions"] = [sns_topic_arn]
        
        self.cloudwatch.put_metric_alarm(**alarm_config)
        logger.info(f"Alarm created: {alarm_name}")
    
    def get_endpoint_status(self) -> Dict:
        """Get current endpoint status."""
        response = self.sagemaker.describe_endpoint(
            EndpointName=self.endpoint_name
        )
        
        return {
            "endpoint_name": response["EndpointName"],
            "status": response["EndpointStatus"],
            "creation_time": str(response.get("CreationTime", "")),
            "last_modified": str(response.get("LastModifiedTime", "")),
        }

