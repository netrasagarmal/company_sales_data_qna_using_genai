import pandas as pd
import io
import json
import os
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI  # Or your preferred model
from langchain.output_parsers import PydanticOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent

# Import the necessary module
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()


# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))


# Global state
global_dataframes: List[pd.DataFrame] = [df_forcast]
global_summaries: List[Dict[str, Any]] = []



def dataframe_basic_info():
    """
    Summarizes the last DataFrame in the provided list.
    
    Args:
        global_dataframes (list): A list of pandas DataFrames.
    
    Returns:
        str: A formatted string containing df.info, df.head(5), and df.columns and top unique value counts for object columns.
    """
    if not global_dataframes or not isinstance(global_dataframes[-1], pd.DataFrame):
        return "No valid DataFrame found in the list."

    df = global_dataframes[-1]

    # Capture df.info() as string
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Capture df.head(5) and df.columns
    head_str = df.head(5).to_string(index=True)
    columns_str = ", ".join(df.columns.astype(str))

    # Top unique values for object columns
    unique_value_summary = "\n=== Top Unique Values in Object Columns ===\n"
    obj_cols = df.select_dtypes(include='object').columns

    if not obj_cols.empty:
        for col in obj_cols:
            value_counts = df[col].value_counts(dropna=False).head(10)
            unique_value_summary += f"\nUnique Values in Column: {col}\n"
            for val, count in value_counts.items():
                unique_value_summary += f"  {repr(val)}: {count}\n"
    else:
        unique_value_summary += "No object columns found.\n"

    # Combine all parts
    summary = (
        "=== DataFrame Info ===\n" + info_str +
        "\n\n=== Column Names ===\n" + columns_str +
        "\n === Top unique values for object columns ===\n" + unique_value_summary +
        "\n=== Head (First 5 Rows) ===\n" + head_str
    )

    return summary

# Define output schema
class EDAQuestionsOutput(BaseModel):
    questions: List[str] = Field(..., description="A list of EDA-related questions")

# Create parser
parser = PydanticOutputParser(pydantic_object=EDAQuestionsOutput)

# Use a regular template with Jinja-style placeholders
prompt_template_str = """
You are a senior data analyst. Given the following comprehensive information about a DataFrame:

{basic_info}

Generate 10-15 high-quality questions that would help perform thorough exploratory data analysis.

Focus your questions on:
1. Statistical Properties (e.g., distributions, correlations, aggregates)
2. Business Insights (e.g., KPIs, trends, segments)
3. Advanced Analysis (e.g., feature engineering, modeling ideas)

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(prompt_template_str)

# Global list to store generated questions
generated_eda_questions: List[str] = []

def generate_questions(basic_info: str) -> List[str]:
    global generated_eda_questions

    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
    result: EDAQuestionsOutput = chain.invoke({"basic_info": basic_info})

    generated_eda_questions = result.questions
    return result.questions



# Create the pandas agent
pandas_agent = create_pandas_dataframe_agent(llm, df_forcast, verbose=True, allow_dangerous_code=True)

# Function to answer each EDA question and return markdown
def generate_question_answer(questions: List[str]) -> str:
    markdown_report = "# EDA Questions and Answers\n\n"

    for i, question in enumerate(questions, 1):
        try:
            if i == 4:
                break
            else:
                response = pandas_agent.invoke({question})
                markdown_report += f"### {i}. {question}\n\n{response}\n\n"
        except Exception as e:
            markdown_report += f"### {i}. {question}\n\n❌ Error: {str(e)}\n\n"

    return markdown_report
