"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Read a required environment variable, raising a clear error if missing."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(
            f"{key} is not set. Copy .env.example to .env and add your key."
        )
    return value


class Settings:
    """Central config. Reads from .env / environment."""

    openai_api_key: str = _require("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    graphiti_model: str = os.getenv("GRAPHITI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # The judge should be at least as capable as the model under evaluation,
    # not weaker -- evaluating gpt-5-mini outputs with a gpt-4o-mini judge is
    # backwards. Defaults to gpt-5-mini; override via JUDGE_MODEL.
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-5-mini")
    # Cheap auxiliary model for the query-rewriter pass (reference resolution
    # only -- doesn't need the main reasoning model's capability).
    query_rewriter_model: str = os.getenv("QUERY_REWRITER_MODEL", "gpt-4o-mini")

    # Neo4j connection (via docker-compose)
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    conversations_db_path: str = os.getenv("CONVERSATIONS_DB_PATH", "data/conversations.db")
    index_checkpoint_path: str = os.getenv("INDEX_CHECKPOINT_PATH", "data/index_checkpoint.json")
    eval_results_path: str = os.getenv("EVAL_RESULTS_PATH", "data/eval_results.json")
    dataset_path: str = os.getenv("DATASET_PATH", "data/convfinqa_dataset.json")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


settings = Settings()
