from langchain.tools import tool
from langchain_tavily import TavilySearch

tavily_search_tool = TavilySearch(max_results=5, topic="general")

@tool("website_search", description="search on web using tavily search for the latest information.")
def website_search(query: str):
    """Up-to-date web info via Tavily"""
    try:
        result = tavily_search_tool.invoke({"query": query})

        # Extract and format the results from Tavily response
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")

            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"
