from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with support for environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # GitHub Configuration
    github_token: Optional[str] = Field(None, description="GitHub personal access token")
    
    # Google APIs Configuration
    google_application_credentials: Optional[str] = Field(
        None, description="Path to Google service account JSON"
    )
    google_oauth_client_id: Optional[str] = None
    google_oauth_client_secret: Optional[str] = None
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_default_region: str = "us-east-1"
    
    # Fly.io Configuration
    fly_api_token: Optional[str] = None
    
    # LLM API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_ai_api_key: Optional[str] = None
    
    # MCP Server Configuration
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 3000
    mcp_log_level: str = "INFO"
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent
    
    def validate_github(self) -> bool:
        """Check if GitHub configuration is valid."""
        return bool(self.github_token)
    
    def validate_google(self) -> bool:
        """Check if Google configuration is valid."""
        return bool(self.google_application_credentials or (
            self.google_oauth_client_id and self.google_oauth_client_secret
        ))
    
    def validate_aws(self) -> bool:
        """Check if AWS configuration is valid."""
        return bool(self.aws_access_key_id and self.aws_secret_access_key)
    
    def validate_fly(self) -> bool:
        """Check if Fly.io configuration is valid."""
        return bool(self.fly_api_token)


# Global settings instance
settings = Settings()