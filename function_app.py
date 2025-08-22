from openai import embeddings
import azure.functions as func
from azure_functions_openapi.decorator import openapi
from azure_functions_openapi.openapi import get_openapi_json
from azure_functions_openapi.swagger_ui import render_swagger_ui
import datetime
import json
import logging
from models import SearchRequest, SearchByIdRequest, SearchResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,  # ensure handlers added even if something pre-configured
)

from search import AzureAISearch
from config import load_search_config
import base64
import struct

logging.getLogger("azure.functions").setLevel(logging.INFO)

logging.info("Startup logger level=%s", logging.getLogger().level)

app = func.FunctionApp()

@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="search",
    description="Search for documents using a query.",
    toolProperties=json.dumps([{
            "propertyName": "query",
            "propertyType": "string",
            "description": "The search query to use for finding documents.",
        }]),
    )

@app.embeddings_input(arg_name="embeddings", input="{arguments.query}", input_type="rawText", embeddings_model="%EMBEDDING_MODEL_DEPLOYMENT_NAME%")
async def search_tool(context: str, embeddings: str) -> dict:
    """Search for documents using a query."""
    try:
        mcp_data = json.loads(context)
        args = mcp_data.get("arguments", {})
        query = args.get("query", "")
        if not query:
            return func.HttpResponse("Missing query parameter", status_code=400)

        # Load configuration
        search_config = load_search_config()
        # Perform search
        ai_search = AzureAISearch(search_config)
        await ai_search.initialize()

        embeddings_data = json.loads(embeddings)
        embedding_vector = embeddings_data["response"]["data"][0]["embedding"]
        results = await ai_search.search_documents(query, embedding_vector, 10)
        
        results_dict = {
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "architecture_url": doc.architecture_url,
                    "content": doc.content,
                    "score": doc.score
                }
                for doc in results.documents
            ]
        }
        return json.dumps(results_dict)

    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        return json.dumps({"error": str(e)})


@app.generic_trigger(
    arg_name="context",
    type="mcpToolTrigger",
    toolName="search_by_doc_id",
    description="Search for documents using the document id.",
    toolProperties=json.dumps([{
            "propertyName": "doc_id",
            "propertyType": "string",
            "description": "The document id to use for finding documents.",
        }]),
    )
async def search_tool_by_doc_id(context: str) -> dict:
    """Search for documents using a query."""
    try:
        mcp_data = json.loads(context)
        args = mcp_data.get("arguments", {})
        doc_id = args.get("doc_id", "")
        if not doc_id:
            return func.HttpResponse("Missing document id parameter", status_code=400)

        # Load configuration
        search_config = load_search_config()
        # Perform search
        ai_search = AzureAISearch(search_config)
        await ai_search.initialize()

        results = await ai_search.get_document_by_id(doc_id)
        results_dict = {
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "architecture_url": doc.architecture_url,
                    "content": doc.content,
                    "score": doc.score
                }
                for doc in results.documents
            ]
        }
        return json.dumps(results_dict)

    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        return json.dumps({"error": str(e)})
    

@app.route(route="http_search", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
@app.embeddings_input(arg_name="embeddings", input="{query}", input_type="rawText", embeddings_model="%EMBEDDING_MODEL_DEPLOYMENT_NAME%")
@openapi(
    summary="Search for documents",
    description="Searches documents using a natural language query with embeddings for semantic relevance.",
    tags=["Search"],
    operation_id="http_search",
    route="/api/http_search",
    method="post",
    request_model=SearchRequest,
    response_model=SearchResponse,
    response={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def http_search(req: func.HttpRequest, embeddings: str) -> func.HttpResponse:
    """HTTP search endpoint."""
    req_body = req.get_json()
    context = json.dumps({
        "arguments": {
            "query": req_body.get("query", "")
        }
    })
    response_json = await search_tool(context, embeddings)
    return func.HttpResponse(response_json, mimetype="application/json")


@app.route(route="http_search_by_doc_id", methods=["POST"], auth_level=func.AuthLevel.FUNCTION)
@openapi(
    summary="Get a document by its ID",
    description="Fetches a specific document by its unique identifier.",
    tags=["Search"],
    operation_id="http_search_by_doc_id",
    route="/api/http_search_by_doc_id",
    method="post",
    request_model=SearchByIdRequest,
    response_model=SearchResponse,
    response={
        200: {"description": "Document retrieved successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def http_search_by_doc_id(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP search by document id endpoint."""
    req_body = req.get_json()
    context = json.dumps({
        "arguments": {
            "doc_id": req_body.get("doc_id", "")
        }
    })
    response_json = await search_tool_by_doc_id(context)
    return func.HttpResponse(response_json, mimetype="application/json")

@app.route(route="openapi.json", auth_level=func.AuthLevel.ANONYMOUS)
def openapi_spec(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(get_openapi_json())

@app.route(route="docs", auth_level=func.AuthLevel.ANONYMOUS)
def swagger_ui(req: func.HttpRequest) -> func.HttpResponse:
    return render_swagger_ui()