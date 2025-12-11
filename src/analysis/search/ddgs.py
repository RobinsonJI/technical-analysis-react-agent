from typing import Dict
import logging
logger = logging.getLogger(__name__)

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

from src.data.basemodels import WebSearch

@tool(args_schema=WebSearch)
def internet_search(query: str, news: bool = False) -> Dict:
    """Searches the internet using Tavily."""
    logger.info(f"Running internet search with the following parameters: \n\t query: '{query}' \n\t news: {news}")

    if news:
        search = DuckDuckGoSearchResults(output_format="list", backend="news")
    else:
        search = DuckDuckGoSearchResults(output_format="list")


    results = search.invoke(query)

    search_results_string = ""

    for result in results:
        search_results_string += f"Source: '{result['link']}' \n"
        search_results_string += f"Title: '{result['title']}' \n"
        search_results_string += f"Title: '{result['snippet']}' \n\n"

    logger.info(f"Internet search results: {search_results_string}")

    return {
        "search_results" : search_results_string,
        "metadata" : {
            "query" : query,
            "news" : news
        }
    }
