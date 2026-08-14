import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# Define Custom Callback Handler
class LoggingCallback(BaseCallbackHandler):
    """Hooks into LangChain events to print raw operational logs."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n🟢 [CALLBACK: LLM CALL STARTED]")

    def on_llm_end(self, response, **kwargs):
        print("🟢 [CALLBACK: LLM RESPONDED]")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'Unknown Tool')
        print(f"\n🔧 [CALLBACK: TOOL EXECUTION STARTED] -> {tool_name}")

    def on_tool_end(self, output, **kwargs):
        print(f"🔧 [CALLBACK: TOOL COMPLETED] -> Output: {str(output)[:80]}...")

# Set up LLM & Tools
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

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

tools = [calculator, get_date]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that uses tools to answer questions accurately."
)

if __name__ == "__main__":
    query_input = {"messages": [{"role": "user", "content": "What is 15% of 2847, and what day is today?"}]}

    print("Executing Agent with Callbacks Enabled...\n")
    
    # Notice: Callbacks attached via the config dictionary at invoke time!
    result = agent.invoke(
        query_input,
        config={"callbacks": [LoggingCallback()]}
    )