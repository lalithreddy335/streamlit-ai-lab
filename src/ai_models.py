"""
AI Model Integration Module
Handles interactions with various AI models
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    name: str
    provider: ModelProvider
    max_tokens: int = 2000
    temperature: float = 0.7
    top_p: float = 0.9


class AIModel:
    """Base class for AI model interactions"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def stream(
        self,
        prompt: str,
        **kwargs
    ):
        """
        Stream response from prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
        
        Yields:
            Response chunks
        """
        raise NotImplementedError("Subclasses must implement stream()")


class OpenAIModel(AIModel):
    """OpenAI model wrapper"""
    
    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.api_key = api_key
        try:
            import openai
            openai.api_key = api_key
            self.client = openai
        except ImportError:
            self.logger.error("openai package not installed")
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: User prompt
            system_message: System message for context
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Placeholder - would use actual OpenAI API
            response = f"Response from {self.config.name}: {prompt[:50]}..."
            return response
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise


class AnthropicModel(AIModel):
    """Anthropic Claude model wrapper"""
    
    def __init__(self, api_key: str, config: ModelConfig):
        super().__init__(config)
        self.api_key = api_key
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response using Anthropic API
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        try:
            # Placeholder - would use actual Anthropic API
            response = f"Claude response: {prompt[:50]}..."
            return response
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise


class ModelFactory:
    """Factory for creating AI model instances"""
    
    _models: Dict[str, type] = {
        ModelProvider.OPENAI.value: OpenAIModel,
        ModelProvider.ANTHROPIC.value: AnthropicModel,
    }
    
    @classmethod
    def create(
        cls,
        config: ModelConfig,
        api_key: Optional[str] = None,
        **kwargs
    ) -> AIModel:
        """
        Create AI model instance
        
        Args:
            config: Model configuration
            api_key: API key for the provider
            **kwargs: Additional arguments
        
        Returns:
            AI model instance
        
        Raises:
            ValueError: If provider not supported
        """
        provider_value = config.provider.value
        
        if provider_value not in cls._models:
            raise ValueError(f"Unsupported model provider: {provider_value}")
        
        model_class = cls._models[provider_value]
        
        if api_key:
            return model_class(api_key, config)
        else:
            return model_class(config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model providers"""
        return list(cls._models.keys())


class ConversationManager:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add message to conversation"""
        self.messages.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.messages.copy()
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def get_context(self) -> str:
        """Get conversation context as string"""
        context = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages
        )
        return context
