import os
from src.graph.state import AgentState


def route_after_router(state: AgentState) -> str:
    """
    Reads the 'intent' from the state and determines the next node to execute.
    """
    intent = state.get("intent")
    
    print(f"--> [Edge] Routing intent '{intent}' to appropriate node...")
    
    if intent == "search":
        # Direct to the node responsible for finding and scraping articles
        return "scraper_node"
    
    elif intent == "config":
        # Direct to the node that updates user preferences in the database
        return "config_node"
    
    else:
        # For 'chat' intent, or any unhandled cases, we can route to a simple chat node
        return "chat_node"


def route_after_filter(state: AgentState) -> str:
    """
    Conditional edge after filter_node.
    Routes to analyzer_node if enough articles passed the filter.
    Routes back to scraper_node if below the minimum threshold, up to a max retry count.
    """
    filtered = state.get("filtered_articles", [])
    seen_urls = state.get("seen_urls", [])
    min_articles = int(os.environ.get("MIN_FILTERED_ARTICLES", 5))
    max_retries = 3

    if len(filtered) >= min_articles:
        print(f"--> [Edge] Filter passed ({len(filtered)} articles). Proceeding to analysis.")
        return "analyzer_node"

    retry_count = len(seen_urls) // int(os.environ.get("MAX_SEARCH_RESULTS", 5))
    if retry_count < max_retries:
        print(f"--> [Edge] Only {len(filtered)} article(s) passed filter (min={min_articles}). Retrying scrape (attempt {retry_count + 1}/{max_retries})...")
        return "scraper_node"

    print(f"--> [Edge] Max retries reached. Proceeding with {len(filtered)} article(s).")
    return "analyzer_node"