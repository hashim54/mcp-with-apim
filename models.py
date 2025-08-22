from typing import Optional
from pydantic import BaseModel, Field, field_validator

class Document(BaseModel):
    """Document model for Azure Search"""
    id: str = Field(..., description="Unique identifier for the document")
    name: str = Field(..., description="Name of the document")
    content: str = Field(..., description="Content of the document")
    architecture_url: str = Field(..., description="URL to the architecture diagram")
    score: float = Field(..., description="Search score for the document")

class SearchResponse(BaseModel):
    """Response model for search results"""
    documents: list[Document] = Field(..., description="List of documents matching the search query")

# NEW: request models
class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")

class SearchByIdRequest(BaseModel):
    doc_id: str = Field(..., description="The document ID to retrieve")