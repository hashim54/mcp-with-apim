import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class AzureSearchConfig(BaseModel):
    """Configuration for Azure Search"""
    endpoint: str = Field(..., description="Azure Search service endpoint")
    key: str = Field(..., description="Azure Search admin key")
    index_name: str = Field(..., description="Azure Search index name")
    


def load_search_config() -> AzureSearchConfig:
    """Load configuration from environment variables with validation"""
    
    # Azure Search config
    search_config = AzureSearchConfig(
        endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT", ""),
        key=os.getenv("AZURE_AI_SEARCH_KEY", ""),
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "")
    )

    return search_config