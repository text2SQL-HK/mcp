# client_demo.py
import asyncio
from fastmcp.client import Client
from mcp.types import TextContent

BASE_URL = "http://127.0.0.1:8000/sse"


async def main() -> None:
    async with Client(BASE_URL) as client:
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])

        result = await client.call_tool("echo", {"text": "안녕하세요 FastMCP!"})

        print("Tool response:")
        for block in result.content:
            if isinstance(block, TextContent):
                print(block.text)
            else:
                print(block)


if __name__ == "__main__":
    asyncio.run(main())
