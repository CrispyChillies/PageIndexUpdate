from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PageIndex Document API"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = Field(
        default="postgresql+psycopg://pageindex:pageindex@localhost:5432/pageindex"
    )
    retrieval_model: str = "gpt-5-mini"
    upload_dir: Path = Path("data/uploads")
    default_model: str = "gpt-4o-2024-11-20"
    toc_check_pages: int = 20
    max_pages_per_node: int = 10
    max_tokens_per_node: int = 20000
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "yes"
    if_add_doc_description: str = "yes"
    if_add_node_text: str = "no"
    if_thinning: str = "no"
    thinning_threshold: int = 5000
    summary_token_threshold: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
