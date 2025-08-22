import logging
import os
from typing import Optional, List
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from config import AzureSearchConfig
from models import Document, SearchResponse
import asyncio
from config import load_search_config
import base64
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureAISearch:
    """Azure Search client for MCP server"""

    def __init__(self, search_config: AzureSearchConfig):
        self._search_client: Optional[SearchClient] = None
        self.search_config = search_config
        self.num_search_results = 15
        self.k_nearest_neighbors = 30
    
    async def initialize(self):
        """Initialize Azure Search"""
        await self._initialize_search_client()
        #await self._initialize_embeddings_client()
    
    async def _initialize_search_client(self):
        """Initialize Azure AI Search client"""
        if self._search_client is None:
            logger.info("Initializing Azure AI Search client...")
            # Use stored config instead of environment variables
            endpoint = self.search_config.endpoint
            key = self.search_config.key
            index_name = self.search_config.index_name
            
            try:
                credential = AzureKeyCredential(key)
                self._search_client = SearchClient(
                    endpoint=endpoint,
                    index_name=index_name,
                    credential=credential
                )
                logger.info("✅ Azure AI Search client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Azure AI Search client: {str(e)}")
                raise
    
    # async def _initialize_embeddings_client(self):
    #     """Initialize Azure OpenAI client for the embeddings model"""
    #     if self._embeddings_client is None:
    #         logger.info("Initializing Azure OpenAI embeddings client...")
            
    #         endpoint = self.azure_openai_embedding_config.endpoint
    #         key = self.azure_openai_embedding_config.key
    #         print(f"endpoint: {endpoint}, key: {key}")
            
    #         try:
    #             self._embeddings_client = AzureOpenAI(
    #                 api_version="2024-12-01-preview",
    #                 azure_endpoint=endpoint,
    #                 api_key=key
    #                 )

    #             logger.info("✅ Azure OpenAI embeddings client initialized successfully")
    #         except Exception as e:
    #             logger.error(f"❌ Failed to initialize Azure OpenAI embeddings client: {str(e)}")
    #             raise

    async def search_documents(self, query: str, query_vector: str, max_results: Optional[int] = None):
        """Search documents using hybrid search (text + vector)"""
        if max_results is None:
            max_results = self.num_search_results
        
        logger.info(f"Searching documents for query: '{query}'")
        
        # Generate vector embedding for the query
        #embedding_response = self._embeddings_client.embeddings.create(input=query,model=self.azure_openai_embedding_config.embeddings_model_name)
        #logger.info(f"Generated vector embedding (dimension: {len(query_vector)})")
        vector_bytes = base64.b64decode(query_vector)
        num_floats = len(vector_bytes) // 4
        vector_floats = list(struct.unpack(f'{num_floats}f', vector_bytes))
        # Create vector query
        vector_queries = [
            VectorizedQuery(
                vector=vector_floats,
                k_nearest_neighbors=self.k_nearest_neighbors,
                fields="content_vector"
            )
        ]
        
        # Perform hybrid search
        results = self._search_client.search(
            search_text=query,
            vector_queries=vector_queries,
            select=["id", "name", "architecture_url", "content"],
            top=max_results
        )
        
        # Process results
        documents = []
        for result in results:
            content_parts = []
            
            if result.get("name"):
                content_parts.append(f"=== NAME ===\n{result['name']}\n=== END NAME ===")
            
            if result.get("architecture_url"):
                content_parts.append(f"=== URL ===\n{result['architecture_url']}\n=== END URL ===")
            
            if result.get("content"):
                content_parts.append(f"=== CONTENT ===\n{result['content']}\n=== END CONTENT ===")
            
            combined_content = "\n\n".join(content_parts)
            
            document = Document(
                id=result["id"],
                name=result["name"],
                architecture_url=result.get("architecture_url", ""),
                content=combined_content,
                score=result["@search.score"]
            )
            documents.append(document)
        
        logger.info(f"Found {len(documents)} documents")
        return SearchResponse(documents=documents)
    

    async def get_document_by_id(self, doc_id: str) -> SearchResponse:
        """Retrieve a single document by its id (primary key) without vector search.

        Args:
            doc_id: The document's primary key value in the Azure AI Search index.

        Returns:
            SearchResponse containing zero or one Document.
        """
        if not doc_id:
            logger.warning("get_document_by_id called with empty id")
            return SearchResponse(documents=[])

        if self._search_client is None:
            # Ensure initialized
            await self._initialize_search_client()

        logger.info(f"Retrieving document by id: {doc_id}")
        try:
            raw = self._search_client.get_document(key=doc_id)

        except Exception as e:
            logger.error(f"Error retrieving document '{doc_id}': {e}")
            raise

        content_parts = []
        if raw.get("name"):
            content_parts.append(f"=== NAME ===\n{raw['name']}\n=== END NAME ===")
        if raw.get("architecture_url"):
            content_parts.append(f"=== URL ===\n{raw['architecture_url']}\n=== END URL ===")
        if raw.get("content"):
            content_parts.append(f"=== CONTENT ===\n{raw['content']}\n=== END CONTENT ===")

        combined_content = "\n\n".join(content_parts)

        document = Document(
            id=raw["id"],
            name=raw.get("name", ""),
            architecture_url=raw.get("architecture_url", ""),
            content=combined_content,
            score=1.0  # Direct get_document has no relevance score; assign nominal value.
        )
        return SearchResponse(documents=[document])

async def main():
    """Main entry point for testing the Azure Search client"""
    # Load configuration
    search_config = load_search_config()
    
    # Initialize Azure Search client
    ai_search = AzureAISearch(search_config)
    try:
        await ai_search.initialize()        
        #response = await ai_search.search_documents("example query", "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")
        #id='22837103-5487-4775-a6a0-3bc8d4179ed4'
        response = await ai_search.get_document_by_id(doc_id='22837103-5487-4775-a6a0-3bc8d4179ed4')
        print(response)
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    # Load configuration
    asyncio.run(main())