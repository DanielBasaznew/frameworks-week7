from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Step A: Define the Pydantic model
class ResumeReview(BaseModel):
    score: int = Field(description="An overall rating out of 100")
    strengths: list[str] = Field(description="Top 3 strengths found in the resume")
    weaknesses: list[str] = Field(description="Top 3 areas for improvement")
    summary: str = Field(description="A brief 2-sentence summary of the candidate")

# Step B: Set up the Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

# Step C: Set up the parser
parser = JsonOutputParser(pydantic_object=ResumeReview)

# Step D: Build the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical HR resume reviewer. Evaluate the resume accurately.\n{format_instructions}"),
    ("user", "Here is the resume text:\n{resume_text}")
])

# Step E: Build the LCEL chain
chain = prompt | llm | parser

# Run the chain on a sample resume
if __name__ == "__main__":
    sample_resume = """
    Alex Mercer
    Software Developer | 2 years experience
    - Built internal Python utilities for log parsing.
    - Managed basic PostgreSQL queries and schema updates.
    - Automated daily file transfers using bash and cron jobs.
    Skills: Python, SQL, Linux, Git.
    """

    print("Running chain...\n")
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "resume_text": sample_resume
    })

    print("Returned Type:", type(result))
    print("Parsed Output:", result)