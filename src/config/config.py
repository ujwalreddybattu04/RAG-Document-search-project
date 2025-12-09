"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class Config:
    """Configuration class for RAG system"""

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    # Use OpenRouter model (Gemma 3 12B free)
    LLM_MODEL = "openai/gpt-4o-mini"

    # OpenAI-compatible API endpoint via OpenRouter
    API_BASE = "https://openrouter.ai/api/v1"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):

        if not cls.OPENROUTER_API_KEY:
            raise ValueError("❌ OPENROUTER_API_KEY not found in .env")

        # Set env vars required by OpenAI-compatible clients
        os.environ["OPENAI_API_KEY"] = cls.OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = cls.API_BASE

        # provider MUST be openai (OpenRouter uses OpenAI-compatible API)
        return init_chat_model(
            model=cls.LLM_MODEL,
            model_provider="openai",
        )
