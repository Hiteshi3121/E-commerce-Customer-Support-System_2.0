from fastmcp import FastMCP
from backend.mcp_server.rag_tool import search_company_docs

mcp = FastMCP("NovaCart MCP Server")


@mcp.tool()
def hello_tool(name: str) -> str:
    """Test tool"""
    return f"Hello {name}, MCP server is running!"


@mcp.tool()
def company_docs_search(query: str) -> str:
    """Search NovaCart company documentation"""
    return search_company_docs(query)


if __name__ == "__main__":
    mcp.run(transport="http")