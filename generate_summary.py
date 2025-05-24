from abc import ABC, abstractmethod
import io
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import json
import os
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.chat_models import ChatOpenAI
# from langchain.output_parsers import PydanticOutputParser
# from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()




# # Global state
# global_dataframes: List[pd.DataFrame] = []
# global_summaries: List[Dict[str, Any]] = []

# --- Abstract Base Class ---
class GenerateFileSummary(ABC):
    """
    Abstract base class for generating file summaries and EDA questions.
    This class defines the interface for generating summaries and questions from a DataFrame.
    """
    def __init__(self, llm = None):
        self.llm = llm

    @abstractmethod
    def generate_summary(self, df:pd.DataFrame = None, file_name: str = None, file_format: str = None) -> Tuple[str, List[str], str]:
        """Generate and return a summary in markdown string format."""
        pass


# --- Main Concrete Class ---
class DataFrameSummaryGenerator(GenerateFileSummary):
    """
    A concrete implementation of GenerateFileSummary that generates summaries and EDA questions from a DataFrame.
    This class uses a language model to analyze the DataFrame and generate insights.
    """

    def __init__(self, llm = None):
        super().__init__(llm=llm)
        self.llm = llm
        # self.pandas_agent = create_pandas_dataframe_agent(llm, self.df, verbose=True, allow_dangerous_code=True)

    @staticmethod
    def dataframe_basic_info(df: pd.DataFrame) -> str:
        """
        Summarizes the last DataFrame in the provided list.
        
        Args:
            global_dataframes (list): A list of pandas DataFrames.
        
        Returns:
            str: A formatted string containing df.info, df.head(5), and df.columns and top unique value counts for object columns.
        """

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

    def generate_summary(self, df:pd.DataFrame, file_name: str = None, file_format: str = None) -> Tuple[str, List[str], str]:
        try:

            # Step 1: Generate basic summary info
            basic_info: str = self.dataframe_basic_info(df=df)

            # 1. Define the response schema fields
            response_schemas = [
                ResponseSchema(name="summary", description="Markdown summary of the DataFrame", type="string"),
                ResponseSchema(name="questions", description="List of 10 EDA questions", type="list", items_type="string")
            ]

            # 2. Create the parser
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

            # 3. Get format instructions
            format_instructions = output_parser.get_format_instructions()

            prompt_template = """
            You are a senior data analyst. Given the following information about a DataFrame:

            {basic_info}

            Generate:
            1. A summary of what is the dataframe about (i.e. domain) and what data it contains.
            2. A list of 10 high-quality EDA questions. Number the questions from 1 to 10.

            Focus your questions on:
            - Statistical Properties
            - Business Insights
            - Advanced Analysis

            Return your answer in **format** as shown below:

            {format_instructions}
            """
    
            prompt = ChatPromptTemplate.from_template(prompt_template)

            # # 6. Use your model
            # llm = ChatOpenAI(temperature=0)  # or use your LLM instance

            chain = prompt | self.llm | output_parser

            # 7. Call the chain
            result = chain.invoke({"basic_info": basic_info, "format_instructions" : format_instructions})

            print("\n\n Result:", result, "Type:", type(result))

            # Result is parsed
            summary_str = result["summary"] if "summary" in result else "No summary generated."
            questions_list = result["questions"] if "questions" in result else []

            questions_str = "\n".join(questions_list) if questions_list else "No questions generated."

            final_summary = """## File Summary:\n\n##### File Name: {}\n##### File Format: {}\n##### Description:\n{}\n\n## Here are some questions that can be asked for EDA:\n{}.""".format(
                file_name if file_name else "N/A",
                file_format if file_format else "N/A",
                summary_str,
                questions_str
            )

            return (summary_str, questions_list, final_summary)
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Handle the error gracefully, return empty strings or raise an exception
            # You can also log the error if needed
            raise 
            # Optionally, you can raise an exception or log the error
            # raise e
            # Or return empty values
        # return ("", [], "")
    

if __name__ == "__main__":
    # Example usage

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

    df_forcast : pd.DataFrame = pd.read_excel('G:/company_sales_genai/company_sales_data_qna_using_genai/data/forcast.xlsx')
    # global_dataframes.append(df_forcast)
    df_summary_generator: GenerateFileSummary = DataFrameSummaryGenerator(llm=llm)
    summary = df_summary_generator.generate_summary(df=df_forcast, file_name="forcast.xlsx", file_format="xlsx")
    # Print the summary
    print("Summary String:")
    print(summary[0])  # summary_str
    print("\nEDA Questions:")
    print(summary[1])  # questions_list
    print("\nFinal Summary:")
    print(summary[2])

    # Define the file path (you can change the name or path as needed)
    file_path = "./output_summary.txt"

    # Write the text to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(summary[2])

    # Return the file path (optional)
    print(f"Text written to: {file_path}")