#!/usr/bin/env python3
"""
Simple MCP Hub server without FastMCP to avoid TaskGroup issues.
"""

import asyncio
import logging
import sys
import os
from typing import Any, Dict, List, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from tools.github_tools_v2 import GitHubTools
from tools.aws_tools import AWSTools
from lib.llm_collaboration import LLMOrchestrator, LLMProvider
from config.settings import settings

# Configure logging - send to stderr to avoid interfering with JSON responses
logging.basicConfig(
    level=getattr(logging, settings.mcp_log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True
)
logger = logging.getLogger(__name__)

# Global orchestrator for LLM collaboration
orchestrator = LLMOrchestrator()

# Create server instance
server = Server(
    name="mcp-hub",
    version="0.1.0"
)

# Tool storage
tools = {}


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name=name,
            description=tool["description"],
            inputSchema=tool["inputSchema"]
        )
        for name, tool in tools.items()
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Optional[Dict[str, Any]] = None
) -> List[types.TextContent]:
    """Execute a tool."""
    if name not in tools:
        raise ValueError(f"Unknown tool: {name}")
    
    try:
        logger.info(f"Executing tool: {name} with arguments: {arguments}")
        handler = tools[name]["handler"]
        result = await handler(arguments or {})
        logger.info(f"Tool {name} returned result of type {type(result)} with length {len(str(result))}")
        
        if result is None or str(result).strip() == "":
            logger.warning(f"Tool {name} returned empty result!")
            return [types.TextContent(type="text", text="Tool executed but returned no output")]
        
        return [types.TextContent(type="text", text=str(result))]
    except Exception as e:
        logger.error(f"Tool {name} error: {e}", exc_info=True)
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


def register_base_tools():
    """Register base tools."""
    
    async def health_check(params: Dict[str, Any]) -> str:
        """Check server health status."""
        try:
            session_count = len(orchestrator.sessions)
            tool_count = len(tools)
            return f"✓ MCP Hub healthy - {tool_count} tools, {session_count} sessions"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return f"✗ Health check failed: {str(e)}"
    
    tools["health_check"] = {
        "description": "Check server health status",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": health_check
    }
    
    async def add_numbers(params: Dict[str, Any]) -> str:
        """Add two numbers together."""
        try:
            a = params.get("a", 0)
            b = params.get("b", 0)
            result = a + b
            return f"{a} + {b} = {result}"
        except Exception as e:
            logger.error(f"Add numbers failed: {e}")
            return f"Error adding numbers: {str(e)}"
    
    tools["add_numbers"] = {
        "description": "Add two numbers together",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        "handler": add_numbers
    }
    
    async def ask_gemini(params: Dict[str, Any]) -> str:
        """Ask Google AI (Gemini) a question."""
        prompt = params.get("prompt")
        model = params.get("model", "gemini-2.0-flash")
        
        logger.info(f"Gemini called with: prompt='{prompt[:50]}...', model={model}")
        
        if not prompt:
            return "Error: prompt is required"
        
        if not settings.google_ai_api_key or settings.google_ai_api_key == "your_google_ai_api_key":
            return "Error: Google AI API key not configured. Please check your .env file."
        
        try:
            logger.info(f"Creating GoogleAI client with model: {model}")
            
            # Create Google AI client directly
            from lib.llm_collaboration import GoogleAIClient, Message
            
            client = GoogleAIClient(model=model)
            logger.info(f"Client created successfully")
            
            messages = [Message(role="user", content=prompt)]
            logger.info(f"Messages created: {len(messages)}")
            
            logger.info(f"Sending request to {model}...")
            response = await client.complete(messages)
            
            logger.info(f"Got response: {len(response.content)} characters")
            
            result = f"[GEMINI {model.upper()}]\n{response.content}"
            
            if response.usage:
                result += f"\n\nTokens used: {response.usage.get('total_tokens', 'unknown')}"
            
            logger.info(f"Returning result with {len(result)} characters")
            return result
            
        except Exception as e:
            error_msg = f"Gemini error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    tools["ask_gemini"] = {
        "description": "Ask Google AI (Gemini) a question",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Question or prompt to send to Gemini"},
                "model": {
                    "type": "string", 
                    "description": "Gemini model to use",
                    "enum": [
                        "gemini-2.5-flash",
                        "gemini-2.0-flash", 
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-flash"
                    ],
                    "default": "gemini-2.0-flash"
                }
            },
            "required": ["prompt"]
        },
        "handler": ask_gemini
    }
    
    logger.info("Base tools registered")


def register_github_tools():
    """Register GitHub integration tools."""
    if not settings.validate_github():
        logger.warning("GitHub tools not loaded - missing configuration")
        return
    
    try:
        github_tools = GitHubTools()
        
        async def github_create_issue(params: Dict[str, Any]) -> str:
            """Create a new GitHub issue."""
            tool = github_tools.create_issue_tool()
            return await tool.handler(params)
        
        tools["github_create_issue"] = {
            "description": "Create a new GitHub issue",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository name (owner/repo)"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body"},
                    "assignees": {"type": "array", "items": {"type": "string"}},
                    "labels": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["repo", "title"]
            },
            "handler": github_create_issue
        }
        
        async def github_create_pr(params: Dict[str, Any]) -> str:
            """Create a new GitHub pull request."""
            tool = github_tools.create_pr_tool()
            return await tool.handler(params)
        
        tools["github_create_pr"] = {
            "description": "Create a new GitHub pull request",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository name (owner/repo)"},
                    "title": {"type": "string", "description": "PR title"},
                    "head": {"type": "string", "description": "Head branch"},
                    "body": {"type": "string", "description": "PR body"},
                    "base": {"type": "string", "description": "Base branch", "default": "main"},
                    "draft": {"type": "boolean", "description": "Draft PR", "default": False}
                },
                "required": ["repo", "title", "head"]
            },
            "handler": github_create_pr
        }
        
        async def github_list_issues(params: Dict[str, Any]) -> str:
            """List GitHub issues with filters."""
            tool = github_tools.list_issues_tool()
            return await tool.handler(params)
        
        tools["github_list_issues"] = {
            "description": "List GitHub issues with filters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository name (owner/repo)"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
                    "assignee": {"type": "string", "description": "Filter by assignee"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 30}
                },
                "required": ["repo"]
            },
            "handler": github_list_issues
        }
        
        logger.info("GitHub tools registered successfully")
        
    except Exception as e:
        logger.error(f"Failed to register GitHub tools: {e}", exc_info=True)


def register_aws_tools():
    """Register AWS integration tools."""
    if not settings.validate_aws():
        logger.warning("AWS tools not loaded - missing configuration")
        return
    
    try:
        aws_tools = AWSTools()
        
        async def aws_ec2_list_instances(params: Dict[str, Any]) -> str:
            """List EC2 instances with optional filters."""
            tool = aws_tools.ec2_list_instances_tool()
            return await tool.handler(params)
        
        tools["aws_ec2_list_instances"] = {
            "description": "List EC2 instances with optional filters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "Instance state filter"},
                    "tag_filters": {"type": "object", "description": "Tag filters"},
                    "region": {"type": "string", "description": "AWS region"}
                },
                "required": []
            },
            "handler": aws_ec2_list_instances
        }
        
        async def aws_s3_list_buckets(params: Dict[str, Any]) -> str:
            """List S3 buckets in the account."""
            tool = aws_tools.s3_list_buckets_tool()
            return await tool.handler(params)
        
        tools["aws_s3_list_buckets"] = {
            "description": "List S3 buckets in the account",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prefix": {"type": "string", "description": "Bucket name prefix"},
                    "region": {"type": "string", "description": "AWS region"}
                },
                "required": []
            },
            "handler": aws_s3_list_buckets
        }
        
        async def aws_s3_list_objects(params: Dict[str, Any]) -> str:
            """List objects in an S3 bucket."""
            tool = aws_tools.s3_list_objects_tool()
            return await tool.handler(params)
        
        tools["aws_s3_list_objects"] = {
            "description": "List objects in an S3 bucket",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bucket": {"type": "string", "description": "Bucket name"},
                    "prefix": {"type": "string", "description": "Object key prefix"},
                    "max_keys": {"type": "integer", "default": 100},
                    "region": {"type": "string", "description": "AWS region"}
                },
                "required": ["bucket"]
            },
            "handler": aws_s3_list_objects
        }
        
        logger.info("AWS tools registered successfully")
        
    except Exception as e:
        logger.error(f"Failed to register AWS tools: {e}", exc_info=True)


async def main():
    """Main entry point."""
    logger.info("Starting MCP Hub server v0.1.0 (simple)")
    logger.info(f"Server host: {settings.mcp_server_host}")
    logger.info(f"Server port: {settings.mcp_server_port}")
    logger.info(f"Log level: {settings.mcp_log_level}")
    
    # Debug: Check API key availability
    logger.info(f"OpenAI API key configured: {bool(settings.openai_api_key and settings.openai_api_key != 'your_openai_api_key')}")
    logger.info(f"Anthropic API key configured: {bool(settings.anthropic_api_key and settings.anthropic_api_key != 'your_anthropic_api_key')}")
    logger.info(f"Google AI API key configured: {bool(settings.google_ai_api_key and settings.google_ai_api_key != 'your_google_ai_api_key')}")
    
    # Register all tools
    register_base_tools()
    register_github_tools()
    register_aws_tools()
    
    logger.info(f"Initialized MCP Hub with {len(tools)} tools")
    
    # Run the server
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-hub",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("MCP Hub server shutting down")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)