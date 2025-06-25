#!/usr/bin/env python3
"""Test script to verify MCP Hub server functionality."""

import asyncio
import logging
from servers.base_v2 import MCPHub
from mcp import Tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_server():
    """Test the MCP Hub server."""
    
    # Create server instance
    hub = MCPHub("test-hub", "0.1.0")
    
    # Create a simple test tool
    async def hello_world(params: dict) -> str:
        name = params.get("name", "World")
        return f"Hello, {name}!"
    
    test_tool = Tool(
        name="hello",
        description="Say hello to someone",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"}
            }
        },
        handler=hello_world
    )
    
    # Register the tool
    hub.register_tool(test_tool)
    
    # Test tool listing
    logger.info(f"Registered tools: {list(hub.tools.keys())}")
    
    # Test tool execution
    result = await hub.tools["hello"].handler({"name": "MCP"})
    logger.info(f"Tool execution result: {result}")
    
    print("\n✓ MCP Hub server test successful!")
    print("\nTo run the full server with all integrations:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Run: python main_v2.py")


if __name__ == "__main__":
    asyncio.run(test_server())