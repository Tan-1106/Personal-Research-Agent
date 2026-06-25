import os
from typing                                     import Dict, Any, List
from pydantic                                   import BaseModel, Field
from concurrent.futures                         import ThreadPoolExecutor
from src.graph.state                            import AgentState
from src.utils.scraper_tools                    import scrape_text_from_url
from langchain_core.messages                    import SystemMessage, HumanMessage
from langchain_google_genai                     import ChatGoogleGenerativeAI
from langchain_tavily                           import TavilySearch
from langchain_core.messages                    import AIMessage


def scraper_node(state: AgentState) -> Dict[str, Any]:
    """
    Searches the internet using Tavily API based on user input and scrapes the content.
    Applies followed/blocked preferences intelligently and resets state between queries.
    """
    print("--> [Scraper Node] Initiating search and scrape sequence...")

    user_input = state.get("user_input", "")
    user_prefs = state.get("user_preferences", {})
    followed = user_prefs.get("followed", [])
    blocked = user_prefs.get("blocked", [])

    # Load previously visited URLs for this query's retry loop.
    seen_urls = state.get("seen_urls", [])

    is_first_round = len(seen_urls) == 0
    if is_first_round:
        print("    - First round detected. Resetting accumulated state from previous query.")

    search_query = user_input
    if followed:
        relevance_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
        relevance_result = relevance_llm.invoke(
            f"The user has these preferred topics: {', '.join(followed)}.\n"
            f"User query: \"{user_input}\"\n"
            f"Is this query a general/open-ended request that should be narrowed down using "
            f"the preferred topics? Or does the user have a specific subject in mind that is "
            f"unrelated to the preferred topics?\n"
            f"Respond with ONLY one word: 'general' or 'specific'."
        )
        query_type = relevance_result.content.strip().lower()
        print(f"    - Query type detected: '{query_type}'")

        if query_type == "general":
            search_query += f" (focus on: {', '.join(followed)})"
            print(f"    - Appending followed topics to query.")
        else:
            print(f"    - Specific query detected. Skipping followed topics to preserve intent.")

    if blocked:
        search_query += f" (exclude: {', '.join(blocked)})"

    # Translate the full search intent to English to retrieve diverse global sources,
    # regardless of the user's preferred display language.
    print("    - Translating query to English for global sourcing...")
    translation_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    translation_result = translation_llm.invoke(
        f"Translate the following search intent into a concise English search query. "
        f"Return ONLY the English query string, with no extra explanation or quotes:\n{search_query}"
    )
    eng_query = translation_result.content.strip()

    # 1. Search Phase using Tavily
    print(f"    - Original input   : '{user_input}'")
    print(f"    - Global query (EN): '{eng_query}'")

    # Escalate max_results on each retry so the search goes deeper every loop iteration.
    base_max = int(os.environ.get("MAX_SEARCH_RESULTS", 5))
    current_max = base_max + len(seen_urls)
    search_tool = TavilySearch(max_results=current_max)

    try:
        # Perform the search using the translated English query
        search_results = search_tool.invoke({"query": eng_query})
        all_found_urls = [res["url"] for res in search_results.get("results", [])]
    except Exception as e:
        print(f"    - Search failed: {e}")
        all_found_urls = []

    # Exclude any URLs that were already scraped in a previous loop iteration.
    new_urls = [u for u in all_found_urls if u not in seen_urls]
    updated_seen_urls = seen_urls + new_urls
    print(f"    - Found {len(new_urls)} new URLs (skipped {len(all_found_urls) - len(new_urls)} already seen).")

    # 2. Scrape Phase
    raw_articles = []
    for url in new_urls:
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
        "urls": new_urls,
        "seen_urls": updated_seen_urls,
        "raw_articles": raw_articles,
        **({"filtered_articles": []} if is_first_round else {})
    }
    
    
class ArticleEvaluation(BaseModel):
    # 1. Define the Pydantic schema for the LLM's evaluation output
    is_relevant: bool = Field(
        description="True if the article is highly relevant to the user's intent and contains substantial information. False if it is clickbait, overly promotional, or irrelevant."
    )
    reason: str = Field(
        description="A short, 1-sentence explanation for why it was kept or discarded."
    )
 

class UserPreferences(BaseModel):
    topics_to_follow: List[str] = Field(
        description="List of topics the user wants to prioritize or follow."
    )
    topics_to_block: List[str] = Field(
        description="List of topics the user explicitly wants to avoid or block."
    )
    preferred_language: str = Field(
        description="The full spelling of the preferred language (e.g., 'Vietnamese', NOT 'vi').",
        default="English"
    )
    confirmation_message: str = Field(
        description="A polite, brief confirmation message (1 sentence) written in the user's preferred language acknowledging the update."
    )
    system_messages: Dict[str, str] = Field(
        description=(
            "Translate these exact English keys into the user's preferred language. "
            "Keys must be: "
            "'thinking' (translating '⏳ Thinking and researching... This might take a minute.'), "
            "'error' (translating '❌ An error occurred. Please try again.'), "
            "'morning' (translating '🌅 Good morning! Preparing your daily digest...'), "
            "'no_results' (translating 'No relevant information found based on your request.'), "
            "'generation_failed' (translating 'Error occurred while generating the final digest.'), "
            "'connection_error' (translating 'Sorry, I am having trouble connecting right now.'), "
            "'lbl_followed' (translating 'Followed'), "
            "'lbl_blocked' (translating 'Blocked'), "
            "'lbl_language' (translating 'Language'), "
            "'lbl_none' (translating 'None')"
        )
    )
    

def filter_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates scraped articles using a fast LLM to remove noise and clickbait.
    """
    print("--> [Filter Node] Starting intelligent noise filtering...")
    
    user_input = state.get("user_input", "")
    
    # Retrieve previously accumulated good articles from prior loops
    previously_filtered = state.get("filtered_articles", [])
    
    # Retrieve the raw articles passed down from the current Scraper Node round
    raw_articles = state.get("raw_articles", []) 
    
    if not raw_articles:
        print("    - No new raw articles to filter in this round.")
        return {"filtered_articles": previously_filtered}

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
        
    # Inject preferences into the system prompt
    user_prefs = state.get("user_preferences", {})
    followed = user_prefs.get("followed", [])
    blocked = user_prefs.get("blocked", [])
    
    system_prompt = raw_prompt.format(
        user_input=user_input,
        followed=", ".join(followed) if followed else "None",
        blocked=", ".join(blocked) if blocked else "None"
    )

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
    
    # Accumulate the newly accepted articles with the ones from previous rounds
    accumulated_articles = previously_filtered + truly_filtered_articles
    print(f"--> [Filter Node] Kept {len(truly_filtered_articles)} out of {len(raw_articles)} new articles. Total accumulated: {len(accumulated_articles)}.")
    
    # 4. Overwrite the state with the combined list
    return {"filtered_articles": accumulated_articles}


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
    Synthesizes individual analyzed reports into a concise, on-topic final digest
    that directly answers the user's original question.
    """
    print("--> [Editor Node] Synthesizing reports into final digest...")
    
    analyzed_reports = state.get("analyzed_reports", [])
    user_input = state.get("user_input", "")
    
    # Fetch translated system messages for fallback responses
    user_prefs = state.get("user_preferences", {})
    language = user_prefs.get("language", "English")
    sys_msgs = user_prefs.get("system_messages", {})
    
    if not analyzed_reports:
        print("    - No reports to edit.")
        return {"final_digest": sys_msgs.get("no_results", "No relevant information found based on your request.")}
        
    # Combine all individual reports into a single string for the LLM
    combined_reports = "\n\n".join(analyzed_reports)
    
    # The temperature is slightly higher (0.4) to allow for more creative writing
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4
    )
    
    prompt_path = os.path.join("src", "prompt", "editor_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        raw_prompt = file.read()
        
    # Inject both the preferred language and the user's original question into the prompt
    system_prompt = raw_prompt.format(language=language, user_input=user_input)
        
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
        final_digest = sys_msgs.get("generation_failed", "Error occurred while generating the final digest.")
        
    # Update state with the final synthesized report
    return {"final_digest": final_digest}


def config_node(state: AgentState) -> Dict[str, Any]:
    """
    Updates the user's long-term research preferences (follows, blocks, language).
    """
    print("--> [Config Node] Updating user preferences...")
    user_input = state.get("user_input", "")
    current_prefs = state.get("user_preferences", {})
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    structured_llm = llm.with_structured_output(UserPreferences)
    
    system_prompt = (
        "Analyze the user's message regarding their research preferences. "
        f"Their CURRENT preferences are: {current_prefs}. "
        "IMPORTANT RULES:\n"
        "1. VIEW ONLY: If the user is simply asking to see/view their current preferences, return the EXACT SAME topics and language. Do NOT clear them.\n"
        "2. ADD: If the user wants to add new topics, COMBINE them with the current topics.\n"
        "3. REMOVE: If the user wants to remove specific topics, delete them from the current list.\n"
        "4. CLEAR ALL: Only return an empty list if the user explicitly asks to clear, reset, or delete all topics.\n"
        "Generate a short confirmation_message (in the preferred language) acknowledging what was done (e.g., 'Here are your current settings' or 'Your preferences have been updated')."
    )
    
    messages = [
        SystemMessage(content=system_prompt), 
        HumanMessage(content=user_input)
    ]
    
    try:
        result = structured_llm.invoke(messages)
        new_prefs = {
            "followed": result.topics_to_follow,
            "blocked": result.topics_to_block,
            "language": result.preferred_language,
            "system_messages": result.system_messages
        }
        sys_msgs = result.system_messages
        lbl_followed = sys_msgs.get("lbl_followed", "Followed")
        lbl_blocked = sys_msgs.get("lbl_blocked", "Blocked")
        lbl_lang = sys_msgs.get("lbl_language", "Language")
        lbl_none = sys_msgs.get("lbl_none", "None")
        
        str_followed = ", ".join(new_prefs["followed"]) if new_prefs["followed"] else lbl_none
        str_blocked = ", ".join(new_prefs["blocked"]) if new_prefs["blocked"] else lbl_none
        
        # Assemble the formatted message deterministically
        msg = (
            f"⚙️ {result.confirmation_message}\n\n"
            f"✅ {lbl_followed}: {str_followed}\n"
            f"❌ {lbl_blocked}: {str_blocked}\n"
            f"🗣 {lbl_lang}: {new_prefs['language'].title()}"
        )
    except Exception as e:
        print(f"--> [Config Node] Error updating config: {e}")
        new_prefs = current_prefs
        sys_msgs = current_prefs.get("system_messages", {})
        msg = sys_msgs.get("error", "❌ An error occurred while updating your preferences.")
        
    return {"user_preferences": new_prefs, "final_digest": msg}


def chat_node(state: AgentState) -> Dict[str, Any]:
    """
    Handles casual conversations with conversation history memory.
    """
    print("--> [Chat Node] Handling chat with memory...")
    user_input = state.get("user_input", "")
    
    user_prefs = state.get("user_preferences", {})
    language = user_prefs.get("language", "English")
    
    # Retrieve the chat history from the state
    chat_history = state.get("chat_history", [])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)
    
    # 1. Initialize the System Prompt
    messages = [
        SystemMessage(content=f"You are a helpful Personal Research Agent. Briefly and politely answer the user's casual chat message. You MUST reply in this language: {language}")
    ]
    
    # 2. Inject previous chat history (limit to the last 10 messages to save tokens)
    for msg in chat_history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    # 3. Append the newest user input
    messages.append(HumanMessage(content=user_input))
    
    try:
        result = llm.invoke(messages)
        reply = result.content
    except Exception as e:
        sys_msgs = user_prefs.get("system_messages", {})
        reply = sys_msgs.get("connection_error", "Sorry, I am having trouble connecting right now.")
        
    # 4. Save the updated history (including the current question and bot response)
    new_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply}
    ]
        
    return {"final_digest": reply, "chat_history": new_history}
