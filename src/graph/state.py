from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    Represents the shared state of the LangGraph workflow.
    Every node in the graph can read from and write to this state.
    """
    # The raw message or command sent by the user via Telegram
    user_input: str
    
    # Classified intent of the user (e.g., 'search', 'config', 'chat')
    intent: str
    
    # Long-term preferences updated by the user (e.g., favorite topics, blocked keywords)
    user_preferences: Dict[str, Any]
    
    # List of URLs extracted from the current scraping round
    urls: List[str]
    
    # Accumulates all URLs visited across retry loops to prevent re-scraping
    seen_urls: List[str]
    
    # Raw articles collected from scraping
    raw_articles: List[Dict[str, str]]
    
    # Filtered, high-quality content ready for deep analysis
    filtered_articles: List[Dict[str, str]]
    
    # Summarized and analyzed insights from each individual source (Map stage)
    analyzed_reports: List[str]
    
    # The final consolidated editorial report sent back to the user
    final_digest: str