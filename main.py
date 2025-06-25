#!/usr/bin/env python3
import asyncio
import logging
from typing import Any, Dict

from mcp import Tool
from mcp.types import TextContent
from pydantic import BaseModel, Field

from servers.base import MCPHub
from tools.github_tools import GitHubTools
from tools.aws_tools import AWSTools
from lib.llm_collaboration import LLMOrchestrator, LLMProvider, Message
from config.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.mcp_log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMCollaborateParams(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    prompt: str = Field(..., description="Prompt to send to LLMs")
    mode: str = Field("parallel", description="Execution mode: parallel or sequential")
    providers: list[str] = Field(None, description="Specific providers for sequential mode")


class MCPHubServer(MCPHub):
    """Extended MCP server with all integrations."""
    
    def __init__(self):
        super().__init__("mcp-hub")
        self.orchestrator = LLMOrchestrator()
        
    async def initialize(self):
        """Initialize the server and load all tools."""
        await super().initialize()
        
        # Load GitHub tools if configured
        if settings.validate_github():
            try:
                github_tools = GitHubTools()
                for tool in github_tools.get_all_tools():
                    self.register_tool(tool)
                logger.info("GitHub tools loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load GitHub tools: {e}")
        
        # Load AWS tools if configured
        if settings.validate_aws():
            try:
                aws_tools = AWSTools()
                for tool in aws_tools.get_all_tools():
                    self.register_tool(tool)
                logger.info("AWS tools loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load AWS tools: {e}")
        
        # Register LLM collaboration tool
        self.register_tool(self._create_llm_collaborate_tool())
        
        # TODO: Add Google APIs and Fly.io integrations
        
        logger.info(f"Initialized with {len(self.tools)} tools")
    
    def _create_llm_collaborate_tool(self) -> Tool:
        """Create the LLM collaboration tool."""
        async def llm_collaborate(params: Dict[str, Any]) -> TextContent:
            args = LLMCollaborateParams(**params)
            
            # Get or create session
            session = self.orchestrator.get_session(args.session_id)
            if not session:
                session = self.orchestrator.create_session(args.session_id)
                self.orchestrator.setup_default_llms(session)
            
            try:
                if args.mode == "parallel":
                    responses = await session.run_parallel(args.prompt)
                    result_text = "\n\n".join([
                        f"[{r.provider.value}] {r.model}:\n{r.content}"
                        for r in responses
                    ])
                else:
                    # For sequential mode, use the same prompt for each provider
                    providers = [LLMProvider(p) for p in args.providers] if args.providers else list(session.llm_clients.keys())
                    prompts = [args.prompt] * len(providers)
                    responses = await session.run_sequential(prompts, providers)
                    result_text = "\n\n".join([
                        f"[{r.provider.value}] {r.model}:\n{r.content}"
                        for r in responses
                    ])
                
                return TextContent(type="text", text=result_text)
                
            except Exception as e:
                logger.error(f"LLM collaboration error: {e}")
                return TextContent(type="text", text=f"Error: {str(e)}")
        
        return Tool(
            name="llm_collaborate",
            description="Run prompts across multiple LLMs for collaboration",
            inputSchema=LLMCollaborateParams.model_json_schema(),
            handler=llm_collaborate
        )


async def main():
    """Main entry point."""
    server = MCPHubServer()
    await server.initialize()
    
    # Run the server
    logger.info(f"Starting MCP Hub server on {settings.mcp_server_host}:{settings.mcp_server_port}")
    
    # For MCP, we use stdio transport
    from mcp.server import stdio
    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())