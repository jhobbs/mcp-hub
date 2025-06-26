import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import httpx
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class Message(BaseModel):
    """Represents a message in a conversation."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Generate a completion from the LLM."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        super().__init__(api_key or settings.openai_api_key, model)
        self.base_url = "https://api.openai.com/v1"
    
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [m.model_dump() for m in messages],
                    **kwargs
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                provider=LLMProvider.OPENAI,
                model=self.model,
                usage=data.get("usage")
            )


class AnthropicClient(LLMClient):
    """Anthropic API client."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key or settings.anthropic_api_key, model)
        self.base_url = "https://api.anthropic.com/v1"
    
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        # Convert messages to Anthropic format
        system_msg = next((m.content for m in messages if m.role == "system"), None)
        conversation = [m for m in messages if m.role != "system"]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": m.role, "content": m.content} for m in conversation],
                    "system": system_msg,
                    "max_tokens": kwargs.get("max_tokens", 4096),
                    **{k: v for k, v in kwargs.items() if k != "max_tokens"}
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                content=data["content"][0]["text"],
                provider=LLMProvider.ANTHROPIC,
                model=self.model,
                usage={"total_tokens": data.get("usage", {}).get("input_tokens", 0) + 
                       data.get("usage", {}).get("output_tokens", 0)}
            )


class GoogleAIClient(LLMClient):
    """Google AI (Gemini) API client."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        super().__init__(api_key or settings.google_ai_api_key, model)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def complete(self, messages: List[Message], **kwargs) -> LLMResponse:
        # Convert messages to Google AI format
        contents = []
        for msg in messages:
            role = "user" if msg.role in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": kwargs.get("max_tokens", 4096),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            usage_data = data.get("usageMetadata", {})
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GOOGLE,
                model=self.model,
                usage={
                    "prompt_tokens": usage_data.get("promptTokenCount", 0),
                    "completion_tokens": usage_data.get("candidatesTokenCount", 0),
                    "total_tokens": usage_data.get("totalTokenCount", 0)
                }
            )


class CollaborationSession:
    """Manages a multi-LLM collaboration session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Message] = []
        self.llm_clients: Dict[LLMProvider, LLMClient] = {}
        self.results: List[LLMResponse] = []
    
    def add_llm(self, provider: LLMProvider, client: LLMClient):
        """Add an LLM to the collaboration session."""
        self.llm_clients[provider] = client
        logger.info(f"Added {provider.value} to session {self.session_id}")
    
    def add_message(self, message: Message):
        """Add a message to the conversation history."""
        self.messages.append(message)
    
    async def run_parallel(self, prompt: str, **kwargs) -> List[LLMResponse]:
        """Run the same prompt across all LLMs in parallel."""
        self.add_message(Message(role="user", content=prompt))
        
        tasks = []
        for provider, client in self.llm_clients.items():
            task = client.complete(self.messages, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, result in enumerate(results):
            provider = list(self.llm_clients.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"LLM error from {provider.value}: {result}")
            else:
                logger.info(f"LLM success from {provider.value}: {len(result.content)} chars")
                valid_results.append(result)
                self.results.append(result)
        
        logger.info(f"Parallel execution completed: {len(valid_results)} successful out of {len(results)} total")
        return valid_results
    
    async def run_sequential(self, prompts: List[str], providers: List[LLMProvider], **kwargs) -> List[LLMResponse]:
        """Run different prompts sequentially across specified LLMs."""
        results = []
        
        for prompt, provider in zip(prompts, providers):
            if provider not in self.llm_clients:
                logger.warning(f"Provider {provider} not available in session")
                continue
            
            self.add_message(Message(role="user", content=prompt))
            client = self.llm_clients[provider]
            
            try:
                response = await client.complete(self.messages, **kwargs)
                results.append(response)
                self.results.append(response)
                self.add_message(Message(
                    role="assistant", 
                    content=response.content,
                    metadata={"provider": provider.value, "model": response.model}
                ))
            except Exception as e:
                logger.error(f"Error with {provider}: {e}")
        
        return results
    
    async def consensus(self, prompt: str, threshold: float = 0.7, **kwargs) -> Optional[str]:
        """Get consensus from multiple LLMs on a prompt."""
        responses = await self.run_parallel(prompt, **kwargs)
        
        if not responses:
            return None
        
        # Simple consensus: if majority agree on key points
        # This is a simplified implementation - could be enhanced with NLP
        contents = [r.content for r in responses]
        
        # For now, return the most common response or first one
        return contents[0] if contents else None


class LLMOrchestrator:
    """Orchestrates multiple LLM collaboration sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
    
    def create_session(self, session_id: str) -> CollaborationSession:
        """Create a new collaboration session."""
        session = CollaborationSession(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get an existing session."""
        return self.sessions.get(session_id)
    
    def setup_default_llms(self, session: CollaborationSession):
        """Set up default LLMs based on available API keys."""
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key":
            session.add_llm(LLMProvider.OPENAI, OpenAIClient())
        
        if settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_api_key":
            session.add_llm(LLMProvider.ANTHROPIC, AnthropicClient())
        
        if settings.google_ai_api_key and settings.google_ai_api_key != "your_google_ai_api_key":
            session.add_llm(LLMProvider.GOOGLE, GoogleAIClient())
        
        logger.info(f"Set up {len(session.llm_clients)} LLMs for session {session.session_id}")