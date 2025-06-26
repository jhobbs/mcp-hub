# MCP Hub - AI Tools & Cloud Integration

A Model Context Protocol (MCP) server that provides Google AI (Gemini) integration and cloud service tools.

## Features

- **Google AI (Gemini) Integration**: Ask questions to Google's Gemini models (2.0-flash, 2.5-flash, etc.)
- **GitHub Integration**: Create issues, pull requests, and list repository information
- **AWS Integration**: Manage EC2, S3, Lambda, and other AWS services
- **Health Monitoring**: Built-in health check and utility tools
- **Extensible Architecture**: Easy to add new tool integrations

## Quick Start

### 1. Setup Environment

```bash
cd /home/jason/mcp/mcp-hub

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Configure API Keys

Edit `.env` file with your API keys:
- GitHub personal access token
- AWS credentials  
- Google AI API key
- Other service credentials as needed

### 3. Configure Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

**For WSL (Windows):**
```json
{
  "mcpServers": {
    "mcp-hub": {
      "command": "wsl",
      "args": [
        "-e", 
        "/home/jason/mcp/mcp-hub/venv/bin/python", 
        "/home/jason/mcp/mcp-hub/main.py"
      ]
    }
  }
}
```

**For Linux/Mac:**
```json
{
  "mcpServers": {
    "mcp-hub": {
      "command": "/home/jason/mcp/mcp-hub/venv/bin/python",
      "args": ["/home/jason/mcp/mcp-hub/main.py"]
    }
  }
}
```

**Testing the server manually:**
```bash
# In WSL/Linux terminal
cd /home/jason/mcp/mcp-hub
source venv/bin/activate
python main.py
```

## Available Tools

### Core Tools
- **`ask_gemini`**: Ask Google AI (Gemini) questions
- **`health_check`**: Check server health status
- **`add_numbers`**: Simple math utility for testing

### GitHub Tools  
- **`github_create_issue`**: Create new GitHub issues
- **`github_create_pr`**: Create pull requests
- **`github_list_issues`**: List issues with filters

### AWS Tools
- **`aws_ec2_list_instances`**: List EC2 instances
- **`aws_s3_list_buckets`**: List S3 buckets
- **`aws_s3_list_objects`**: List objects in S3 buckets

## Usage Examples

### Ask Gemini (Primary Feature)
```json
{
  "prompt": "What is the moon made of?",
  "model": "gemini-2.0-flash"
}
```

Available models: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-2.0-flash-exp`, `gemini-1.5-flash`

### Create GitHub Issue
```json
{
  "repo": "username/repository",
  "title": "Bug: API endpoint returns 500",
  "body": "Description of the issue...",
  "labels": ["bug", "high-priority"]
}
```

### List AWS S3 Buckets
```json
{
  "prefix": "my-project",
  "region": "us-east-1"
}
```

## Architecture

```
mcp-hub/
├── main.py              # Main MCP server
├── config/              # Configuration management
│   └── settings.py
├── tools/               # Tool integrations
│   ├── github_tools.py
│   ├── aws_tools.py
│   └── ...
└── lib/                 # Core libraries
    └── llm_collaboration.py  # Google AI client
```

## Adding New Tools

1. Create a new file in `tools/` directory
2. Implement tool classes following the pattern in existing tools
3. Register tools in `main.py` during server initialization

Example tool structure:
```python
async def my_tool(params: Dict[str, Any]) -> str:
    """My custom tool."""
    return "Tool result"

tools["my_tool"] = {
    "description": "Description of what this tool does",
    "inputSchema": {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param"]
    },
    "handler": my_tool
}
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

## License

[Your preferred license]