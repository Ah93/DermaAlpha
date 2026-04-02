"""
config.py — Single source of truth for all application settings.
Reads from .env via python-dotenv. Fails fast with helpful messages if required
keys are missing.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when required environment variables are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    # HuggingFace
    hf_token: str
    hf_model_id: str
    hf_inference_mode: str          # "api" | "ollama" | "local"

    # Ollama (used when hf_inference_mode = "ollama")
    ollama_base_url: str
    ollama_model: str

    # Gemini (used when hf_inference_mode = "gemini")
    gemini_api_key: str
    gemini_model: str

    # Pinecone
    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_index_name: str

    # Embeddings
    embedding_model: str
    embedding_dim: int

    # Application
    max_retrieved_encounters: int
    similarity_threshold: float
    log_level: str
    image_cache_dir: Path
    max_image_dim: int


def load_settings() -> Settings:
    """Load and validate settings from .env file."""
    load_dotenv()

    required = {
        "PINECONE_API_KEY": "Pinecone API key (https://app.pinecone.io)",
    }

    missing = []
    for key, description in required.items():
        if not os.getenv(key):
            missing.append(f"  {key}  —  {description}")

    if missing:
        raise ConfigurationError(
            "Missing required environment variables. "
            "Copy .env.example to .env and fill in the values:\n"
            + "\n".join(missing)
        )

    inference_mode = os.getenv("HF_INFERENCE_MODE", "ollama").lower()
    if inference_mode not in {"api", "ollama", "local", "gemini"}:
        raise ConfigurationError(
            f"HF_INFERENCE_MODE must be 'api', 'ollama', 'local', or 'gemini', got '{inference_mode}'"
        )

    try:
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
        max_retrieved = int(os.getenv("MAX_RETRIEVED_ENCOUNTERS", "5"))
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
        max_image_dim = int(os.getenv("MAX_IMAGE_DIM", "896"))
    except ValueError as e:
        raise ConfigurationError(f"Invalid numeric setting in .env: {e}") from e

    return Settings(
        hf_token=os.getenv("HF_TOKEN", ""),
        hf_model_id=os.getenv("HF_MODEL_ID", "google/medgemma-4b-it"),
        hf_inference_mode=inference_mode,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llava:7b"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "derma-encounters-v2"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO"),
        embedding_dim=embedding_dim,
        max_retrieved_encounters=max_retrieved,
        similarity_threshold=similarity_threshold,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        image_cache_dir=Path(os.getenv("IMAGE_CACHE_DIR", "./image_cache")),
        max_image_dim=max_image_dim,
    )


# Module-level singleton — imported by all other modules
settings = load_settings()
