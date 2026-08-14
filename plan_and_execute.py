import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()
console = Console()

# Initialize LLMs
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

# ==========================================
# 1. TOOLS & EXECUTOR AGENT
# ==========================================
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: a math expression as a string."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Input: the search query."""
    try:
        from ddgs import DDGS
        with DDGS(timeout=5) as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n".join(r['body'] for r in results) if results else "No search results found. Answer based on your existing knowledge."
    except Exception as e:
        return f"Search engine unavailable ({e}). Please answer using your internal knowledge if possible."

tools = [calculator, web_search]

executor_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a precise research assistant executing one step of a multi-step plan. Be factual and concise."
)

# ==========================================
# 2. SCHEMAS (PLANNER & VERIFIER)
# ==========================================
class Plan(BaseModel):
    """Execution plan breaking down the research task."""
    reasoning: str = Field(description="Brief explanation of why this plan was chosen")
    steps: list[str] = Field(description="Ordered list of 3 to 5 distinct, sequential research steps")

class VerificationResult(BaseModel):
    """Quality check on the research results."""
    is_complete: bool = Field(description="True if the collected context fully answers the original goal")
    missing_points: list[str] = Field(default_factory=list, description="Any unanswered questions or missing data")
    final_summary: str = Field(description="A comprehensive synthesis answering the user's objective")

planner_llm = llm.with_structured_output(Plan)
verifier_llm = llm.with_structured_output(VerificationResult)

# ==========================================
# 3. PIPELINE FUNCTIONS
# ==========================================
def execute_step(step: str, context: str) -> str:
    """Run a single step using the agent, injecting accumulated context."""
    prompt = f"Target Step: {step}\n\nContext gathered so far from previous steps:\n{context if context else 'No previous context yet.'}"
    result = executor_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    
    content = result["messages"][-1].content
    
    # SAFETY FIX: If Gemini returns a list of blocks instead of a string, extract the text
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        return content[0].get("text", str(content))
        
    return str(content)




def run_plan_and_execute(goal: str):
    console.print(Panel(f"[bold cyan]User Goal:[/bold cyan] {goal}", title="🎯 Mission Start"))

    # Stage 1: Planning
    with console.status("[bold yellow]Generating execution plan...[/bold yellow]"):
        plan: Plan = planner_llm.invoke(f"Create a clear, 3-4 step execution plan to answer: '{goal}'")
    
    plan_table = Table(title="📋 Generated Plan", show_lines=True)
    plan_table.add_column("Step #", justify="center", style="bold green", width=8)
    plan_table.add_column("Instruction", style="white")
    for i, step in enumerate(plan.steps, 1):
        plan_table.add_row(f"Step {i}", step)
    console.print(plan_table)
    console.print(f"[dim]Planner Reasoning: {plan.reasoning}[/dim]\n")

    # Stage 2: Execution
    accumulated_context = ""
    for i, step in enumerate(plan.steps, 1):
        console.print(f"[bold blue]Executing Step {i}/{len(plan.steps)}:[/bold blue] {step}")
        with console.status(f"[bold green]Running Step {i}...[/bold green]"):
            step_output = execute_step(step, accumulated_context)
        
        console.print(Panel(step_output, title=f"Output: Step {i}", border_style="blue"))
        accumulated_context += f"\n--- Step {i} ({step}) Findings ---\n{step_output}\n"

    # Stage 3: Verification & Synthesis
    console.print("\n[bold magenta]Running Verification & Final Synthesis...[/bold magenta]")
    with console.status("[bold magenta]Evaluating completeness...[/bold magenta]"):
        verify_prompt = (
            f"Original Objective: {goal}\n\n"
            f"Collected Research Findings:\n{accumulated_context}\n\n"
            "Evaluate if the findings fully satisfy the objective, list any missing items, and provide a polished final summary."
        )
        verification: VerificationResult = verifier_llm.invoke(verify_prompt)

    # Stage 4: Results Display
    status_style = "green" if verification.is_complete else "yellow"
    status_text = "COMPLETE" if verification.is_complete else "INCOMPLETE / GAPS IDENTIFIED"
    
    console.print(Panel(
        f"[bold {status_style}]Status: {status_text}[/bold {status_style}]\n"
        f"Missing Points: {', '.join(verification.missing_points) if verification.missing_points else 'None'}\n\n"
        f"[bold white]Final Synthesis:[/bold white]\n{verification.final_summary}",
        title="🏁 Final Verified Result",
        border_style=status_style
    ))

# ==========================================
# 4. EXECUTE TEST QUERY
# ==========================================
if __name__ == "__main__":
    test_query = "Find the current CEO of DeepMind, determine the year they won the Nobel Prize, and calculate how many years elapsed between their founding of DeepMind and that Nobel Prize win."
    run_plan_and_execute(test_query)