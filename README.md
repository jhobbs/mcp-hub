# MCP Hub - Multi-LLM Collaboration Platform

A Model Context Protocol (MCP) server that enables multiple LLMs to collaborate and integrates with various cloud services and APIs for "less attended" development workflows.

## Features

- **Multi-LLM Collaboration**: Run prompts across OpenAI, Anthropic, and Google AI models in parallel or sequential mode
- **GitHub Integration**: Create issues, pull requests, and list repository information
- **AWS Integration**: Manage EC2, S3, Lambda, and other AWS services
- **Google APIs Integration**: Access Drive, Calendar, and other Google services (coming soon)
- **Fly.io Integration**: Deploy and manage applications (coming soon)
- **Extensible Architecture**: Easy to add new tool integrations

## Quick Start

### 1. Clone and Install

```bash
cd /home/jason/mcp/mcp-hub
pip install -e .
```

### 2. Configure Environment

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- GitHub personal access token
- AWS credentials
- LLM API keys (OpenAI, Anthropic, Google AI)
- Other service credentials as needed

### 3. Run the Server

```bash
python main_v2.py
```

Or test the server first:
```bash
python test_server.py
```

The server will start and be available for MCP clients to connect.

## Usage Examples

### Multi-LLM Collaboration

The server provides an `llm_collaborate` tool that allows you to:

1. **Parallel Execution**: Run the same prompt across multiple LLMs simultaneously
```json
{
  "tool": "llm_collaborate",
  "arguments": {
    "session_id": "dev-session-1",
    "prompt": "Generate a Python function to parse JSON with error handling",
    "mode": "parallel"
  }
}
```

2. **Sequential Execution**: Run prompts in sequence, building on previous responses
```json
{
  "tool": "llm_collaborate",
  "arguments": {
    "session_id": "dev-session-1",
    "prompt": "Review and improve this code",
    "mode": "sequential",
    "providers": ["openai", "anthropic"]
  }
}
```

### GitHub Integration

Create issues, PRs, and manage repositories:

```json
{
  "tool": "github_create_issue",
  "arguments": {
    "repo": "username/repository",
    "title": "Bug: API endpoint returns 500",
    "body": "Description of the issue...",
    "labels": ["bug", "high-priority"]
  }
}
```

### AWS Integration

Manage AWS resources:

```json
{
  "tool": "aws_lambda_invoke",
  "arguments": {
    "function_name": "my-data-processor",
    "payload": {"action": "process", "data": "..."}
  }
}
```

## Architecture

```
mcp-hub/
├── servers/          # MCP server implementations
│   └── base.py      # Base MCP server class
├── tools/           # Tool integrations
│   ├── github_tools.py
│   ├── aws_tools.py
│   └── ...
├── lib/             # Core libraries
│   └── llm_collaboration.py
├── config/          # Configuration management
│   └── settings.py
└── main.py          # Entry point
```

## Adding New Integrations

1. Create a new file in `tools/` directory
2. Implement tool classes following the pattern in existing tools
3. Register tools in `main.py` during server initialization

Example tool structure:
```python
from mcp import Tool
from mcp.types import TextContent

class MyServiceTools:
    def create_example_tool(self) -> Tool:
        async def handler(params: Dict[str, Any]) -> TextContent:
            # Tool implementation
            return TextContent(type="text", text="Result")
        
        return Tool(
            name="my_service_action",
            description="Description of what this tool does",
            inputSchema=ParamsModel.model_json_schema(),
            handler=handler
        )
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
ruff check .
```

### Type Checking
```bash
mypy .
```

## Security Notes

- Store all API keys and credentials in environment variables
- Never commit `.env` files to version control
- Use least-privilege IAM roles for AWS access
- Rotate API keys regularly

## Future Enhancements

- Google Home integration for IoT control
- GPU provider integrations for ML workloads
- Enhanced LLM collaboration with semantic analysis
- Workflow automation and scheduling
- Web UI for monitoring and management

## License

[Your preferred license]