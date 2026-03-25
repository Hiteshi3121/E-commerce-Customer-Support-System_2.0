import asyncio
from fastmcp import Client


client = Client("http://127.0.0.1:8000/mcp")


async def search_company_docs(query: str):

    async with client:

        result = await client.call_tool(
            "company_docs_search",
            {"query": query}
        )

        return result.data