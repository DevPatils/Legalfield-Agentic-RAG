"""Central settings. Everything provider-specific is a config flip, never an import
scattered through the codebase -- the embedding/rerank/LLM choices in Architecture.md §4
are alternatives, so nothing downstream should hard-code one.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Providers ---
    embedding_provider: str = "voyage"  # voyage | openai
    voyage_api_key: str = ""
    openai_api_key: str = ""

    # Voyage throttles accounts with no payment method to 3 RPM / 10K TPM (the 200M
    # free tokens still apply -- it is a rate cap, not a volume cap). Adding a card
    # lifts this to the standard limits; raise these to match if you do.
    embed_max_rpm: int = 3
    embed_max_tpm: int = 10_000

    rerank_provider: str = "cohere"  # cohere | local | none
    cohere_api_key: str = ""

    llm_provider: str = "anthropic"  # anthropic | openai
    # Reasoning-heavy nodes: sufficiency check, generation, faithfulness.
    llm_model: str = "claude-sonnet-5"
    # The planner is near-mechanical (classify + split a question), so it runs on a
    # cheaper model. With ~6 LLM calls per query, an eval sweep adds up fast; set this
    # to the same value as llm_model if you would rather not vary it.
    llm_model_fast: str = "claude-haiku-4-5"
    anthropic_api_key: str = ""

    # --- Infrastructure ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_chunks"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "agentic_rag_legal"

    # --- Agent tuning (Architecture.md §7) ---
    max_refine_iterations: int = 3
    retrieve_top_n: int = 40
    rerank_top_k: int = 6
    rrf_k: int = 60

    # --- Paths ---
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def embedding_model(self) -> str:
        return {
            "voyage": "voyage-law-2",
            "openai": "text-embedding-3-large",
        }[self.embedding_provider]

    @property
    def embedding_dim(self) -> int:
        return {"voyage": 1024, "openai": 3072}[self.embedding_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
