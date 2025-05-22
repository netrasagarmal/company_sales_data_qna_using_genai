from abc import ABC, abstractmethod
import io
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()


# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Global state
global_dataframes: List[pd.DataFrame] = []
global_summaries: List[Dict[str, Any]] = []

# --- Abstract Base Class ---
class GenerateFileSummary(ABC):

    @abstractmethod
    def generate_summary(self) -> str:
        """Generate and return a summary in markdown string format."""
        pass


# --- Output Schema for Questions ---
class EDAQuestionsOutput(BaseModel):
    questions: List[str] = Field(..., description="A list of EDA-related questions")


# --- Main Concrete Class ---
class DataFrameSummaryGenerator(GenerateFileSummary):
    generated_eda_questions: List[str] = []

    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.pandas_agent = create_pandas_dataframe_agent(llm, self.df, verbose=True, allow_dangerous_code=True)

    @staticmethod
    def dataframe_basic_info(df: pd.DataFrame) -> str:
        """
        Summarizes the last DataFrame in the provided list.
        
        Args:
            global_dataframes (list): A list of pandas DataFrames.
        
        Returns:
            str: A formatted string containing df.info, df.head(5), and df.columns and top unique value counts for object columns.
        """
        if not global_dataframes or not isinstance(global_dataframes[-1], pd.DataFrame):
            return "No valid DataFrame found in the list."

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

    @classmethod
    def generate_questions(cls, basic_info: str) -> List[str]:

        parser = PydanticOutputParser(pydantic_object=EDAQuestionsOutput)
        
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

        chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
        result: EDAQuestionsOutput = chain.invoke({"basic_info": basic_info})

        cls.generated_eda_questions = result.questions
        return result.questions

    def generate_answer(self, questions: List[str]) -> str:
        markdown_report = "# EDA Questions and Answers\n\n"

        for i, question in enumerate(questions, 1):
            try:
                if i == 4:  # optional limit
                    break
                response = self.pandas_agent.invoke({question})
                markdown_report += f"### {i}. {question}\n\n{response}\n\n"
            except Exception as e:
                markdown_report += f"### {i}. {question}\n\n❌ Error: {str(e)}\n\n"

        return markdown_report

    def generate_summary(self) -> str:
        # Step 1: Generate basic summary info
        basic_info: str = self.dataframe_basic_info(df=self.df)

        # Step 2: Generate EDA questions
        questions: List[str] = self.generate_questions(basic_info=basic_info)

        # Step 3: Use agent to answer questions and compile markdown
        report : str = self.generate_answer(questions=questions)

        return report

# Example usage

df_forcast : pd.DataFrame = pd.read_excel('G:/company_sales_genai/company_sales_data_qna_using_genai/data/forcast.xlsx')
global_dataframes.append(df_forcast)
df_summary_generator: GenerateFileSummary = DataFrameSummaryGenerator(data=global_dataframes[-1])
summary = df_summary_generator.generate_summary()
print(summary)