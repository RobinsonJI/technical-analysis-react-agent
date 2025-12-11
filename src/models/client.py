import os
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler

# Configure logging
logger = logging.getLogger(__name__)

class ModelClient:
    """Client for managing LLM connections with logging support."""
    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            temperature: float = 0.7,
            top_p: float = 0.7,
            seed: float = 0.7,
            max_tokens: Optional[int] = None,
            enable_logging: bool = True,
        ):
        """
        Initialise ModelClient.
        
        Args:
            api_key: API key (defaults to TRADING_API_KEY env var)
            base_url: Base URL for API (defaults to openrouter.ai)
            model: Model name (defaults to openai/gpt-4.1)
            temperature: Temperature for model responses (0-1)
            max_tokens: Maximum tokens in response
            enable_logging: Enable detailed logging of model calls
        """
        self.model = model 
        self.api_key = api_key or os.getenv("TRADING_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.temperature = temperature
        self.seed = seed,
        self.top_p = top_p
        self.enable_logging = enable_logging

        if not self.api_key:
            raise ValueError("TRADING_API_KEY environment variable not set")
        
        self._client = None
        logger.info(f"ModelClient initialised: {self.model}")

    def get_client(self) -> ChatOpenAI:
        """Get or create the LLM client with logging."""
        if self._client is None:
            callbacks = []
            
            # Add logging callback if enabled
            if self.enable_logging:
                callbacks.append(StdOutCallbackHandler())
            
            self._client = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model,
                temperature=self.temperature,
                callbacks=callbacks,
            )
            
            logger.info(f"Created LLM client: {self.model}")
        
        return self._client