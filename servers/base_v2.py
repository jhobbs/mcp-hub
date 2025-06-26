import asyncio
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp import Tool, Resource
import mcp.types as types

logger = logging.getLogger(__name__)


class MCPHub:
    """MCP Hub server that orchestrates multiple tool integrations."""
    
    def __init__(self, name: str = "mcp-hub", version: str = "0.1.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Resource] = {}
        
        # Create the MCP server instance
        self.server = Server(
            name=self.name,
            version=self.version,
            instructions="Multi-LLM collaboration hub with cloud service integrations"
        )
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List all available tools."""
            return [
                types.Tool(
                    name=name,
                    description=tool.description,
                    inputSchema=tool.inputSchema
                )
                for name, tool in self.tools.items()
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]] = None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution."""
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            
            tool = self.tools[name]
            result = await tool.handler(arguments or {})
            
            # Convert result to MCP format
            if isinstance(result, str):
                return [types.TextContent(type="text", text=result)]
            elif hasattr(result, 'text'):
                return [types.TextContent(type="text", text=result.text)]
            else:
                return [types.TextContent(type="text", text=str(result))]
        
        @self.server.list_resources()
        async def handle_list_resources() -> list[types.Resource]:
            """List all available resources."""
            return [
                types.Resource(
                    uri=uri,
                    name=resource.name,
                    description=resource.description,
                    mimeType=resource.mimeType
                )
                for uri, resource in self.resources.items()
            ]
    
    def register_tool(self, tool: Tool):
        """Register a tool with the server."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: Resource):
        """Register a resource with the server."""
        self.resources[resource.uri] = resource
        logger.info(f"Registered resource: {resource.uri}")
    
    async def run(self):
        """Run the MCP server using stdio transport."""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.name,
                    server_version=self.version,
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )