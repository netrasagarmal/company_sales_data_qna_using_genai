# from typing import Annotated, TypedDict
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# import pandas as pd
# import os, io
# from typing import List
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOpenAI

# # Load environment variables from the .env file (if present)
# load_dotenv()


# # Initialize LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# # Define the global dataframe list
# global_dataframes: List[pd.DataFrame] = []
# df_forcast : pd.DataFrame = pd.read_excel('G:/company_sales_genai/company_sales_data_qna_using_genai/data/forcast.xlsx')
# df_wine : pd.DataFrame = pd.read_csv('G:/company_sales_genai/company_sales_data_qna_using_genai/data/wine.csv')
# global_dataframes.append(df_forcast)
# global_dataframes.append(df_wine)

# # Define the state schema
# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     question: str

# # Initialize the graph
# graph_builder = StateGraph(State)

# # Node function that matches expected return type
# def get_answer_from_df(state: State) -> State:
#     # answer = "Answer from DataFrame"  # Placeholder
#     # prompt = """
#     # You are a data analyst. You have two dataframes. Below given is a sequence of dataframes that would be given in input choose dataframe dumber accordingly to answer the question
#     # Dataframe 1. df_forcast: A dataframe containing sales forecast data.
#     # Dataframe 2. df_wine: A dataframe containing wine sales data.
#     # You can use the dataframes to answer questions.
#     # {}
#     # """.format(state["messages"][-1].content)
#     pandas_agent = create_pandas_dataframe_agent(llm=llm, df=[df_forcast,df_wine], verbose=True, allow_dangerous_code=True)
#     response = pandas_agent.invoke({state["messages"][-1].content})
#     # response = pandas_agent.invoke({state["messages"][-1].content})
#     print(response, type(response))
#     return {
#         "messages": state["messages"] + [{"role": "assistant", "content": response["output"]}],
#         "question": state["question"]
#     }

# # Build the graph
# graph_builder.add_node("get_answer_from_df", get_answer_from_df)
# graph_builder.set_entry_point("get_answer_from_df")
# graph_builder.add_edge("get_answer_from_df", END)

# # Compile the graph
# graph = graph_builder.compile()

# # Input
# x = {
#     "messages": [{"role": "user", "content": "give me shape of the forcast dataframe"}],
#     "question": "give me info about the dataframe",
# }

# # Run the graph
# y = graph.invoke(x)
# # print(y, type(y))

from typing import Annotated, TypedDict, List, Tuple, Dict, Union, Optional, Any
from langgraph.graph import StateGraph, END
# from langchain.core.runnable import Runnable
from langgraph.graph.message import add_messages
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
import os, pandas as pd
from dotenv import load_dotenv

# Load .env values
load_dotenv()

# --- Define the Graph State Schema ---
class State(TypedDict):
    messages: Annotated[List, add_messages]

# --- Main Class ---
class DocQA:
    """
    A class to handle document-based question answering using LangChain and Pandas.
    """

    def __init__(self):
        self.df = None
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.agent = None
        self.graph = None


    def load_document(self, file_path: str, file_type: str, file_name: str)->Dict[str, Any]:
        """
        Load a document into the agent.
        Args:
            file_path (str): Path to the document.
            file_type (str): Type of the document ('xlsx' or 'csv').
            file_name (str): Name of the document.
        Returns:
            dict: Status of the loading process.
        """
        try:
            if file_type == "xlsx":
                self.df = pd.read_excel(file_path)
            elif file_type == "csv":
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file type. Please use 'xlsx' or 'csv'.")
            
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                allow_dangerous_code=True
            )
            self.graph = self._build_graph()
        except Exception as e:
            print(f"Error loading document: {e}")
            return {"status":"exception", "message": f"Exception in DocQa, load_document().\n Exception: \n{str(e)}"}
        
        return {"status":"success"}


        
    # Define the node logic
    def get_answer_from_df(self, state: State) -> State:
        """
        Get answer from the DataFrame based on the user's question.
        Args:
            state (State): The current state of the graph.
        Returns:
            State: The updated state with the answer.
        """

        question = state["messages"][-1].content
        
        response = self.agent.invoke({question})
        print(response, type(response))
        print("\n\n\n")
        
        return {
            "messages": state["messages"] + [{"role": "assistant", "content": str(response["output"])}]
        }
    
    def _build_graph(self):
        """
        Build the state graph for the question-answering process.
        Returns:
            StateGraph: The compiled state graph.
        """

        # Create a graph builder
        graph_builder = StateGraph(State)

        # Build graph
        graph_builder.add_node("get_answer_from_df", self.get_answer_from_df)
        graph_builder.set_entry_point("get_answer_from_df")
        graph_builder.add_edge("get_answer_from_df", END)

        return graph_builder.compile()

    def ask(self, question: str) -> Union[str, dict]:
        """
        Ask a question to the agent and get the answer.
        Args:
            question (str): The question to ask.
        Returns:
            str: The answer from the agent.
        """

        # Construct initial state
        state = {
            "messages": [{"role": "user", "content": question}]
        }
        # Run the graph and return final message
        result = self.graph.invoke(state)
        return str(result["messages"][-1].content)
    

if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv("sales_forecast.csv")
    # df_forcast : pd.DataFrame = pd.read_excel('G:/company_sales_genai/company_sales_data_qna_using_genai/data/forcast.xlsx')
    # df_wine : pd.DataFrame = pd.read_csv('G:/company_sales_genai/company_sales_data_qna_using_genai/data/wine.csv')
    # global_dataframes.append(df_forcast)
    # global_dataframes.append(df_wine)

    # df = pd.read_csv("sales_forecast.csv")
    # df_forcast : pd.DataFrame = pd.read_excel('G:/company_sales_genai/company_sales_data_qna_using_genai/data/forcast.xlsx')
    qa_agent = DocQA()
    qa_agent.load_document(file_path='G:/company_sales_genai/company_sales_data_qna_using_genai/data/forcast.xlsx', file_type='xlsx', file_name='forcast.xlsx')

    response = qa_agent.ask("give the list of all columns in the dataframe")
    print(response)
