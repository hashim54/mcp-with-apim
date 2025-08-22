import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv(Path(__file__).resolve().with_name(".env"))


APIM_HOST = os.getenv("APIM_HOST", "")

async def main():
    # Connect to a streamable HTTP server
    async with streamablehttp_client(APIM_HOST) as (read_stream, write_stream, _):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")

            for t in tools.tools:
                try:
                    schema = t.inputSchema if isinstance(t.inputSchema, dict) else t.inputSchema.model_dump()
                except Exception:
                    schema = getattr(t, "inputSchema", None)
                print(f"\nTool: {t.name}\nDescription: {getattr(t,'description','')}\nSchema:\n{json.dumps(schema, indent=2, default=str)}")

            payload = {"ApiHttp_searchPostRequest": {"query": "Azure Service Bus"}}

            result = await session.call_tool("searchForDocuments", payload)

            print(f"Search results: {result}")

if __name__ == "__main__":
    asyncio.run(main())