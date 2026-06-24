import os
from typing                                     import Dict, Any
from pydantic                                   import BaseModel, Field
from concurrent.futures                         import ThreadPoolExecutor
from src.graph.state                            import AgentState
from src.utils.scraper_tools                    import scrape_text_from_url
from langchain_core.messages                    import SystemMessage, HumanMessage
from langchain_google_genai                     import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search    import TavilySearchResults


def scraper_node(state: AgentState) -> Dict[str, Any]:
    """
    Searches the internet using Tavily API based on user input and scrapes the content.
    """
    print("--> [Scraper Node] Initiating search and scrape sequence...")
    
    user_input = state.get("user_input", "")
    
    # 1. Search Phase using Tavily
    print(f"    - Querying internet for: '{user_input}'")
    
    # Initialize Tavily search tool (returns top 3 most relevant results)
    search_tool = TavilySearchResults(max_results=3)
    
    try:
        # Perform the search and extract URLs
        search_results = search_tool.invoke({"query": user_input})
        found_urls = [res["url"] for res in search_results]
    except Exception as e:
        print(f"    - Search failed: {e}")
        found_urls = []
        
    print(f"    - Found {len(found_urls)} potential URLs.")
    
    # 2. Scrape Phase
    raw_articles = []
    for url in found_urls:
        print(f"    - Scraping content from: {url}")
        content = scrape_text_from_url(url)
        
        if not content.startswith("Error"):
            raw_articles.append({
                "url": url,
                "content": content
            })
            print(f"    - Success: Retrieved {len(content)} characters.")
        else:
            print(f"    - Failed: {content}")
            
    print(f"--> [Scraper Node] Completed scraping {len(raw_articles)} articles.")
    
    # 3. Update the AgentState
    return {
        "urls": found_urls,
        "filtered_articles": raw_articles
    }
    
    

class ArticleEvaluation(BaseModel):
    # 1. Define the Pydantic schema for the LLM's evaluation output
    is_relevant: bool = Field(
        description="True if the article is highly relevant to the user's intent and contains substantial information. False if it is clickbait, overly promotional, or irrelevant."
    )
    reason: str = Field(
        description="A short, 1-sentence explanation for why it was kept or discarded."
    )


def filter_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates scraped articles using a fast LLM to remove noise and clickbait.
    """
    print("--> [Filter Node] Starting intelligent noise filtering...")
    
    user_input = state.get("user_input", "")
    
    # Retrieve the raw articles passed down from the Scraper Node
    raw_articles = state.get("filtered_articles", []) 
    
    if not raw_articles:
        print("    - No articles to filter.")
        return {"filtered_articles": []}

    # 2. Initialize the gatekeeper model (Flash-lite for speed and low cost)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0 
    )
    
    # Enforce structured output to get a strict True/False decision
    evaluator = llm.with_structured_output(ArticleEvaluation)
    
    prompt_path = os.path.join("src", "prompt", "filter_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        raw_prompt = file.read()
    system_prompt = raw_prompt.format(user_input=user_input)
    
    truly_filtered_articles = []
    
    # 3. Iterate and evaluate each article
    for article in raw_articles:
        print(f"    - Evaluating: {article['url']}")
        
        # Optimization: We only need to show the LLM the first 1500 characters to judge relevance
        content_preview = article['content'][:1500]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Article Preview:\n{content_preview}")
        ]
        
        try:
            # Let Gemini decide
            result = evaluator.invoke(messages)
            decision_text = "KEEP" if result.is_relevant else "DISCARD"
            print(f"      Decision: {decision_text} - {result.reason}")
            
            if result.is_relevant:
                truly_filtered_articles.append(article)
        except Exception as e:
            print(f"      Evaluation failed: {e}. Keeping article by default.")
            truly_filtered_articles.append(article)
    
    print(f"--> [Filter Node] Kept {len(truly_filtered_articles)} out of {len(raw_articles)} articles.")
    
    # 4. Overwrite the state with only the clean, highly relevant articles
    return {"filtered_articles": truly_filtered_articles}


def analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    Performs deep analysis on the filtered articles in parallel (Map-Reduce).
    Extracts core insights, statistics, and pros/cons.
    """
    print("--> [Analyzer Node] Starting parallel deep analysis...")
    
    filtered_articles = state.get("filtered_articles", [])
    if not filtered_articles:
        print("    - No articles to analyze.")
        return {"analyzed_reports": []}
        
    # We use Gemini 2.5 Flash for best price/performance ratio in reasoning tasks
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )
    
    prompt_path = os.path.join("src", "prompt", "analyzer_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        system_prompt = file.read()
        
    def analyze_single_article(article: Dict[str, str]) -> str:
        """Helper function to analyze a single article."""
        print(f"    - Analyzing: {article['url']}")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"URL: {article['url']}\nContent:\n{article['content']}")
        ]
        try:
            result = llm.invoke(messages)
            return f"Source: {article['url']}\nAnalysis:\n{result.content}\n"
        except Exception as e:
            print(f"      Failed to analyze {article['url']}: {e}")
            return f"Source: {article['url']}\nAnalysis: Failed to analyze.\n"

    # Run the analysis in parallel using ThreadPoolExecutor for Map-Reduce simulation
    with ThreadPoolExecutor(max_workers=5) as executor:
        analyzed_reports = list(executor.map(analyze_single_article, filtered_articles))
        
    print(f"--> [Analyzer Node] Completed analysis of {len(analyzed_reports)} articles.")
    
    # Update state with the compiled reports
    return {"analyzed_reports": analyzed_reports}


def editor_node(state: AgentState) -> Dict[str, Any]:
    """
    Acts as the Chief Editor. Synthesizes individual analyzed reports 
    into a final, cohesive Research Digest.
    """
    print("--> [Editor Node] Synthesizing reports into final digest...")
    
    analyzed_reports = state.get("analyzed_reports", [])
    
    if not analyzed_reports:
        print("    - No reports to edit.")
        return {"final_digest": "No relevant information found based on your request."}
        
    # Combine all individual reports into a single string for the LLM
    combined_reports = "\n\n".join(analyzed_reports)
    
    # We use Gemini 2.5 Pro for the final synthesis to ensure premium writing quality
    # The temperature is slightly higher (0.4) to allow for more creative writing
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.4
    )
    
    prompt_path = os.path.join("src", "prompt", "editor_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        system_prompt = file.read()
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Individual Reports:\n{combined_reports}")
    ]
    
    try:
        result = llm.invoke(messages)
        final_digest = result.content
        print("--> [Editor Node] Final digest created successfully.")
    except Exception as e:
        print(f"--> [Editor Node] Failed to create digest: {e}")
        final_digest = "Error occurred while generating the final digest."
        
    # Update state with the final synthesized report
    return {"final_digest": final_digest}
