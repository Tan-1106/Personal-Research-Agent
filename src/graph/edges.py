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