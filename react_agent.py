import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent # Updated Import!

# Load environment variables
load_dotenv()

# Step A: Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# Step B: Define your Week 3 tools using the @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: a math expression as a string."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_date(query: str = "") -> str:
    """Get the current date and time. Input: any date-related question."""
    return f"Current date: {datetime.datetime.now().strftime('%A, %B %d, %Y')}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Input: the search query."""
    try:
        from ddgs import DDGS # Updated import!
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
        return "\n".join(r['body'] for r in results) if results else "No results found."
    except Exception as e:
        # If it hits a rate limit or error, explicitly tell the LLM so it stops looping!
        return f"Search failed due to an error: {e}. Stop searching and tell the user."

tools = [calculator, get_date, web_search]

# Step C: Create the Agent Engine (Updated for LangChain v1.x)
system_prompt = "You are a helpful assistant that uses tools to answer questions accurately."
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# Step D: Run it and print the trace
if __name__ == "__main__":
    print("Executing Agent Loop...\n")
    
    input_state = {
        "messages": [
            {"role": "user", "content": "How many champions league goal did Cristiano Ronaldo score for Manchester United?"}
        ]
    }
    
    result = agent.invoke(input_state)

    # Print every message in the trace to reveal the ReAct logic
    print("--- TRACE ---")
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__
        
        if msg.content:
            content = msg.content
        elif hasattr(msg, 'tool_calls') and msg.tool_calls:
            content = f"Tool Call Request: {msg.tool_calls}"
        else:
            content = "Empty/System Message"
            
        print(f"[{msg_type}]: {content}\n")