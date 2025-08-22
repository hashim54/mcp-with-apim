import asyncio
import os
from urllib import response
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from azure.core.credentials import AzureKeyCredential
from contextlib import AsyncExitStack
from openai import AzureOpenAI
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Tuple
import logging

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClientwithOpenAI:
    """Test client for the MCP server using FastMCP"""

    def __init__(self, script_path: str):
        self.script_path = script_path
        self.server_params = StdioServerParameters(
            command="python",   # Command to run the MCP server script
            args=[script_path],
            env=None  # Use None to inherit current environment
        )

        self._connected = False # Indicates if the session is connected, used to prevent multiple connections from being initiated for the same class instance
        self._lock = asyncio.Lock()  # Prevents concurrent connect/disconnect calls, once connected, the connect state is used to prevent multiple connections from being initiated for the same class instance

        self._read = None  # Read stream for the MCP client
        self._write = None  # Write stream for the MCP client
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None

        # Initialize OpenAI client once here
        self.openai_client: Optional[AzureOpenAI] = None
        if self.openai_endpoint and self.openai_key:
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.openai_endpoint,
                api_key=self.openai_key, 
                api_version="2024-12-01-preview"
            )
        else:
            raise ValueError("Azure OpenAI endpoint or key is not set in environment variables")

    async def __aenter__(self):
        """Called when entering 'async with' block"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'async with' block"""
        await self.disconnect()

    async def connect(self):
        """Connect to the MCP server and initialize session"""

        async with self._lock:  # Ensure only one coroutine can connect at a time
            if self._connected:
                logger.warning("Session already connected")
                return
            
            try:
                logger.info(f"Connecting to MCP server at {self.script_path}")
                
                # Create exit stack to manage contexts
                self.exit_stack = AsyncExitStack()
                
                # Connect to stdio client
                self._read, self._write = await self.exit_stack.enter_async_context(
                    stdio_client(self.server_params)
                )
                
                # Create and initialize session
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self._read, self._write)
                )
                
                await self.session.initialize()
                self._connected = True
                logger.info("MCP session initialized successfully")
                
                # List available tools
                #tools = await self.session.list_tools()
                #logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {e}")
                await self.disconnect()
                raise

    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None
            self._read = None
            self._write = None
            logger.info("Disconnected from MCP server")


    # async def initialize_test_mcp_server(self):
    #     # Connect to the server

    #     async with stdio_client(self.server_params) as (read, write):
    #         async with ClientSession(read, write) as session:
    #             # Initialize the session
    #             await session.initialize()

    #             tools = await session.list_tools()

    #             print("Available tools:", tools.tools[0].inputSchema)

    #             # Test the add tool
    #             result = await session.call_tool("add", {"a": 5, "b": 3})
    #             print(f"Add result: {result}")
                
    #             # Test the greeting resource
    #             greeting = await session.read_resource("greeting://Alice")
    #             print(f"Greeting: {greeting}")
                
    #             # Test the prompt
    #             prompt = await session.get_prompt("greet_user", {"name": "Bob", "style": "formal"})
    #             print(f"Prompt: {prompt}")
        
    #     return "MCP client test completed successfully"
        

    async def aoai_chat_completion_with_mcp(self, messages: list[dict], model: str = "gpt-4.1") -> dict:
        """Call the Azure OpenAI chat completion API"""

        mcp_tools = await self.session.list_tools()

        openai_tools = []
        for tool in mcp_tools.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # Define the function for the model
        # openai_tool_defn_for_reference = [
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "get_current_time",
        #             "description": "Get the current time in a given location",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "The city name, e.g. San Francisco",
        #                     },
        #                 },
        #                 "required": ["location"],
        #             },
        #         }
        #     }
        # ]

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            max_tokens=1000,
            temperature=0.7
        )

                # Handle function calls if present
        if response.choices[0].message.tool_calls:
            logger.info("Function calls detected in response")
            logger.info(f"Function calls: {response.choices[0].message.tool_calls}")
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the MCP tool
                result = await self.session.call_tool(function_name, function_args)
                logger.info(f"Tool call result: {result}")

        return response.choices[0].message.content if response.choices else ""
            
# Run the test
async def main():
    async with MCPClientwithOpenAI(script_path="mcp-example.py") as mcp_openai_client_test:
        
        await mcp_openai_client_test.aoai_chat_completion_with_mcp(
            messages=[{"role": "user", "content": "Hello, can you add 7 and 89?"}],
            model="gpt-4.1"
        )


if __name__ == "__main__":
   asyncio.run(main())