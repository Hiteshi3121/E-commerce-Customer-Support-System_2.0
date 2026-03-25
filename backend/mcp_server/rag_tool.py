from backend.mcp_server.vectorstore import get_retriever


def search_company_docs(query: str) -> str:
    """
    Retrieve relevant company FAQ documents for a query
    """

    retriever = get_retriever()

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join([doc.page_content for doc in docs])

    return context