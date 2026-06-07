import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Configuration settings for the PRISM system.
    Loads settings from environment variables and an optional .env file.
    """
    # SettingsConfigDict specifies configuration behavior in Pydantic v2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    anthropic_api_key: str = Field(default="mock_key", validation_alias="ANTHROPIC_API_KEY")
    model_name: str = "claude-3-5-sonnet-20241022"  # Primary reasoning model
    max_tokens: int = 2000
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Pipeline config
    fast_filter_top_n: int = 30
    debate_top_n: int = 10
    shortlist_size: int = 10
    
    # Signal weights by role type
    weights: dict = {
        "senior_ic": {
            "capability_fit": 0.30,
            "trajectory": 0.15,
            "impact": 0.25,
            "evidence_confidence": 0.15,
            "hidden_talent": 0.05,
            "behavioral": 0.10
        },
        "engineering_lead": {
            "capability_fit": 0.20,
            "trajectory": 0.15,
            "impact": 0.20,
            "evidence_confidence": 0.10,
            "hidden_talent": 0.05,
            "behavioral": 0.30
        },
        "startup_engineer": {
            "capability_fit": 0.25,
            "trajectory": 0.25,
            "impact": 0.20,
            "evidence_confidence": 0.10,
            "hidden_talent": 0.10,
            "behavioral": 0.10
        }
    }

# Instantiate settings
settings = Settings()

# Self-Test block
if __name__ == "__main__":
    print("[Config] Self-Test Running...")
    print(f"Model Name: {settings.model_name}")
    print(f"API Key Status: {'Configured' if settings.anthropic_api_key and settings.anthropic_api_key != 'mock_key' else 'Using default/mock key'}")
    assert settings.fast_filter_top_n == 30
    assert "senior_ic" in settings.weights
    print("[Config] Self-Test Completed Successfully!")
