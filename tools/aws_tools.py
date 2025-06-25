import logging
from typing import Any, Dict, List, Optional
import boto3
from botocore.exceptions import ClientError
from mcp import Tool
from mcp.types import TextContent
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class AWSToolBase(BaseModel):
    """Base class for AWS tool parameters."""
    region: Optional[str] = Field(None, description="AWS region (uses default if not specified)")


class EC2ListInstancesParams(AWSToolBase):
    state: Optional[str] = Field(None, description="Filter by instance state (running, stopped, etc)")
    tag_filters: Optional[Dict[str, str]] = Field(None, description="Filter by tags")


class S3ListBucketsParams(AWSToolBase):
    prefix: Optional[str] = Field(None, description="Filter buckets by prefix")


class S3ListObjectsParams(AWSToolBase):
    bucket: str = Field(..., description="S3 bucket name")
    prefix: Optional[str] = Field(None, description="Object key prefix")
    max_keys: int = Field(100, description="Maximum number of objects to return")


class LambdaListFunctionsParams(AWSToolBase):
    runtime: Optional[str] = Field(None, description="Filter by runtime (python3.9, nodejs18.x, etc)")


class LambdaInvokeParams(AWSToolBase):
    function_name: str = Field(..., description="Lambda function name or ARN")
    payload: Dict[str, Any] = Field({}, description="JSON payload to send to function")
    invocation_type: str = Field("RequestResponse", description="RequestResponse or Event")


class AWSTools:
    """Collection of AWS integration tools."""
    
    def __init__(self):
        if not settings.validate_aws():
            raise ValueError("AWS credentials not configured")
        
        # Initialize boto3 session
        self.session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_default_region
        )
    
    def ec2_list_instances_tool(self) -> Tool:
        """Create a tool for listing EC2 instances."""
        async def list_instances(params: Dict[str, Any]) -> TextContent:
            args = EC2ListInstancesParams(**params)
            
            try:
                ec2 = self.session.client('ec2', region_name=args.region or settings.aws_default_region)
                
                filters = []
                if args.state:
                    filters.append({'Name': 'instance-state-name', 'Values': [args.state]})
                
                if args.tag_filters:
                    for key, value in args.tag_filters.items():
                        filters.append({'Name': f'tag:{key}', 'Values': [value]})
                
                response = ec2.describe_instances(Filters=filters)
                
                instances = []
                for reservation in response['Reservations']:
                    for instance in reservation['Instances']:
                        instances.append(
                            f"ID: {instance['InstanceId']}, "
                            f"Type: {instance['InstanceType']}, "
                            f"State: {instance['State']['Name']}, "
                            f"IP: {instance.get('PublicIpAddress', 'N/A')}"
                        )
                
                return TextContent(
                    type="text",
                    text="\n".join(instances) if instances else "No instances found"
                )
            except ClientError as e:
                logger.error(f"AWS EC2 error: {e}")
                return TextContent(type="text", text=f"Error: {str(e)}")
        
        return Tool(
            name="aws_ec2_list_instances",
            description="List EC2 instances with optional filters",
            inputSchema=EC2ListInstancesParams.model_json_schema(),
            handler=list_instances
        )
    
    def s3_list_buckets_tool(self) -> Tool:
        """Create a tool for listing S3 buckets."""
        async def list_buckets(params: Dict[str, Any]) -> TextContent:
            args = S3ListBucketsParams(**params)
            
            try:
                s3 = self.session.client('s3', region_name=args.region or settings.aws_default_region)
                response = s3.list_buckets()
                
                buckets = []
                for bucket in response['Buckets']:
                    if args.prefix and not bucket['Name'].startswith(args.prefix):
                        continue
                    buckets.append(
                        f"{bucket['Name']} (created: {bucket['CreationDate'].strftime('%Y-%m-%d')})"
                    )
                
                return TextContent(
                    type="text",
                    text="\n".join(buckets) if buckets else "No buckets found"
                )
            except ClientError as e:
                logger.error(f"AWS S3 error: {e}")
                return TextContent(type="text", text=f"Error: {str(e)}")
        
        return Tool(
            name="aws_s3_list_buckets",
            description="List S3 buckets in the account",
            inputSchema=S3ListBucketsParams.model_json_schema(),
            handler=list_buckets
        )
    
    def s3_list_objects_tool(self) -> Tool:
        """Create a tool for listing S3 objects."""
        async def list_objects(params: Dict[str, Any]) -> TextContent:
            args = S3ListObjectsParams(**params)
            
            try:
                s3 = self.session.client('s3', region_name=args.region or settings.aws_default_region)
                
                paginator = s3.get_paginator('list_objects_v2')
                page_iterator = paginator.paginate(
                    Bucket=args.bucket,
                    Prefix=args.prefix or '',
                    PaginationConfig={'MaxItems': args.max_keys}
                )
                
                objects = []
                for page in page_iterator:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            objects.append(
                                f"{obj['Key']} ({obj['Size']} bytes, "
                                f"modified: {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')})"
                            )
                
                return TextContent(
                    type="text",
                    text="\n".join(objects) if objects else "No objects found"
                )
            except ClientError as e:
                logger.error(f"AWS S3 error: {e}")
                return TextContent(type="text", text=f"Error: {str(e)}")
        
        return Tool(
            name="aws_s3_list_objects",
            description="List objects in an S3 bucket",
            inputSchema=S3ListObjectsParams.model_json_schema(),
            handler=list_objects
        )
    
    def lambda_list_functions_tool(self) -> Tool:
        """Create a tool for listing Lambda functions."""
        async def list_functions(params: Dict[str, Any]) -> TextContent:
            args = LambdaListFunctionsParams(**params)
            
            try:
                lambda_client = self.session.client('lambda', region_name=args.region or settings.aws_default_region)
                
                paginator = lambda_client.get_paginator('list_functions')
                page_iterator = paginator.paginate()
                
                functions = []
                for page in page_iterator:
                    for func in page['Functions']:
                        if args.runtime and func['Runtime'] != args.runtime:
                            continue
                        functions.append(
                            f"{func['FunctionName']} "
                            f"(Runtime: {func['Runtime']}, "
                            f"Memory: {func['MemorySize']}MB, "
                            f"Timeout: {func['Timeout']}s)"
                        )
                
                return TextContent(
                    type="text",
                    text="\n".join(functions) if functions else "No functions found"
                )
            except ClientError as e:
                logger.error(f"AWS Lambda error: {e}")
                return TextContent(type="text", text=f"Error: {str(e)}")
        
        return Tool(
            name="aws_lambda_list_functions",
            description="List Lambda functions with optional runtime filter",
            inputSchema=LambdaListFunctionsParams.model_json_schema(),
            handler=list_functions
        )
    
    def lambda_invoke_tool(self) -> Tool:
        """Create a tool for invoking Lambda functions."""
        async def invoke_function(params: Dict[str, Any]) -> TextContent:
            args = LambdaInvokeParams(**params)
            
            try:
                lambda_client = self.session.client('lambda', region_name=args.region or settings.aws_default_region)
                
                import json
                response = lambda_client.invoke(
                    FunctionName=args.function_name,
                    InvocationType=args.invocation_type,
                    Payload=json.dumps(args.payload)
                )
                
                # Read the response payload
                payload = response['Payload'].read().decode('utf-8')
                
                result = f"Status: {response['StatusCode']}\n"
                if payload:
                    result += f"Response: {payload}"
                
                return TextContent(type="text", text=result)
            except ClientError as e:
                logger.error(f"AWS Lambda error: {e}")
                return TextContent(type="text", text=f"Error: {str(e)}")
        
        return Tool(
            name="aws_lambda_invoke",
            description="Invoke a Lambda function with optional payload",
            inputSchema=LambdaInvokeParams.model_json_schema(),
            handler=invoke_function
        )
    
    def get_all_tools(self) -> List[Tool]:
        """Get all AWS tools."""
        return [
            self.ec2_list_instances_tool(),
            self.s3_list_buckets_tool(),
            self.s3_list_objects_tool(),
            self.lambda_list_functions_tool(),
            self.lambda_invoke_tool()
        ]