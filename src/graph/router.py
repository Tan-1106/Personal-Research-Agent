import os
from src.graph.state            import AgentState
from pydantic                   import BaseModel, Field
from langchain_google_genai     import ChatGoogleGenerativeAI
from langchain_core.messages    import SystemMessage, HumanMessage


# 1. Define the explicit structured output for the LLM
class RouterOutput(BaseModel):
    intent: str = Field(
        description="The classified intent of the user. Must be exactly one of: 'search', 'config', 'chat'."
    )
    
    
def router_node(state: AgentState) -> dict:
    """
    Analyzes the user's input and routes it to the appropriate downstream node.
    """
    user_input = state.get("user_input", "")
    
    # 2. Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0
    )
    
    # Enforce the LLM to reply strictly following the Pydantic schema
    structured_llm = llm.with_structured_output(RouterOutput)
    
    # 3. System prompt for classification logic
    prompt_path = os.path.join("src", "prompt", "router_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as file:
        system_prompt = file.read()
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    print(f"--> [Router Node] Analyzing intent for: '{user_input}'")
    
    # 4. Invoke the model
    result = structured_llm.invoke(messages)
    
    print(f"--> [Router Node] Intent classified as: '{result.intent}'")
    
    # 5. Return the partial state update (this writes to AgentState)
    return {"intent": result.intent}