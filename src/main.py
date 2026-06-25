import os
from src.graph.state                import AgentState
from src.graph.router               import router_node
from src.graph.edges                import route_after_router, route_after_filter
from src.graph.nodes                import scraper_node, filter_node, analyzer_node, editor_node, config_node, chat_node
from langgraph.graph                import StateGraph, START, END


def build_graph(checkpointer=None):
    """
    Constructs the LangGraph workflow topology.
    """
    # 1. Initialize the graph with the AgentState
    workflow = StateGraph(AgentState)
    
    # 2. Add all nodes to the graph
    workflow.add_node("router_node", router_node)
    workflow.add_node("scraper_node", scraper_node)
    workflow.add_node("filter_node", filter_node)
    workflow.add_node("analyzer_node", analyzer_node)
    workflow.add_node("editor_node", editor_node)
    workflow.add_node("config_node", config_node)
    workflow.add_node("chat_node", chat_node)
    
    # 3. Define the edges (The execution flow)
    # The graph always starts at the Router Node
    workflow.add_edge(START, "router_node")
    
    # Router dynamically decides the next node based on 'intent'
    workflow.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "scraper_node": "scraper_node",
            "config_node": "config_node",
            "chat_node": "chat_node"
        }
    )
    
    # The core Research Pipeline: Scrape -> Filter -> conditionally retry or Analyze -> Edit
    workflow.add_edge("scraper_node", "filter_node")
    workflow.add_conditional_edges(
        "filter_node",
        route_after_filter,
        {
            "scraper_node": "scraper_node",  # Loop back if not enough articles
            "analyzer_node": "analyzer_node"  # Proceed when threshold is met
        }
    )
    workflow.add_edge("analyzer_node", "editor_node")
    workflow.add_edge("editor_node", END)
    
    # Config and Chat pipelines go straight to END for now
    workflow.add_edge("config_node", END)
    workflow.add_edge("chat_node", END)
    
    # 4. Compile the graph with dynamic checkpointer
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
    app = workflow.compile(checkpointer=checkpointer)
    return app


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    app = build_graph()
    
    # Test the graph with a sample research query
    test_input = "Find the latest news about Agentic AI workflows"
    print(f"\n[Test] Starting Agent with Input: '{test_input}'\n")
    
    initial_state = {
        "user_input": test_input,
        "intent": "",
        "user_preferences": {},
        "urls": [],
        "filtered_articles": [],
        "analyzed_reports": [],
        "final_digest": ""
    }
    
    # Run the graph and stream the updates from each node
    for output in app.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in output.items():
            print(f"\n--- Output from: {node_name} ---")
            if "final_digest" in state_update:
                print(f"\n========== FINAL RESEARCH DIGEST ==========\n{state_update['final_digest']}\n===========================================\n")