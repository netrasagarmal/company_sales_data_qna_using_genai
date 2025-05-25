from typing import Annotated, TypedDict, List, Tuple, Dict, Union, Optional, Any
from langgraph.graph import StateGraph, END
# from langchain.core.runnable import Runnable
from langgraph.graph.message import add_messages
from langchain.agents.agent import AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
import os, pandas as pd
from dotenv import load_dotenv
from generate_summary import GenerateFileSummary, DataFrameSummaryGenerator
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate

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
        
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.graph = None
        self.files_info: Dict[str, Dict[str,Any]] = {}
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.df_agents: Dict[str, AgentExecutor] = {}
        self.files_knowledge_base: str = "These are the files in the knowledge base: \n\n"
        self.all_files_summary: str = "## Uploaded Files Summary\n\n"
        self.state = {
            "messages": [{"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."}]
        }

        # Build the state graph
        self.graph = self._build_graph()


    def load_document(self, file_paths: str, file_types: str, file_names: str)->Dict[str, Any]:
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
            for file_path, file_type, file_name in zip(file_paths, file_types, file_names):
                file_summary = None
                df = None
                agent = None
                if file_type in ["xlsx", "csv"]:
                    if file_type == "xlsx":
                        df = pd.read_excel(file_path)
                    elif file_type == "csv":
                        df = pd.read_csv(file_path)
                    else:
                        raise ValueError("Unsupported file type. Please use 'xlsx' or 'csv'.")
                    print("\n\n 1 -------------------------------")
                    agent = create_pandas_dataframe_agent(
                        llm=self.llm,
                        df=df,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=3,
                        allow_dangerous_code=True
                    )
                    print("\n\n 2 -------------------------------")
                    df_summary_generator:GenerateFileSummary = DataFrameSummaryGenerator(llm = self.llm)

                    summary, questions, markdown_summary = df_summary_generator.generate_summary(
                        df=df, file_name=file_name, file_format=file_type
                    )
                    print("\n\n 3 -------------------------------")
                    self.files_info[file_name] = {
                        "file_type": file_type,
                        "summary": summary,
                        "questions": questions,
                        "markdown_summary": markdown_summary,
                    }
                    self.dataframes[file_name] = df
                    self.df_agents[file_name] = agent
                    print("\n\n 4 -------------------------------")
                    file_summary = markdown_summary
                    

                    self.files_knowledge_base += f"File Name: {file_name}\nFile Type: {file_type}\nFile Content Description:{summary}\n\n"

                elif file_type == "pdf":
                    pass
                else:
                    raise ValueError("Unsupported file type. Please use 'xlsx', 'csv', or 'pdf'.")
                
                self.all_files_summary = self.all_files_summary + file_summary + "\n\n"
                
        except Exception as e:
            print(f"Error loading document: {e}")
            return {"status":"exception", "message": f"Exception in DocQa, load_document().\n Exception: \n{str(e)}"}
        
        return {"status":"success", "file_summary": self.all_files_summary}

    def decide_df_agent(self, question:str) -> Optional[str]:
        """
        Decide which DataFrame agent to use based on the question.
        Args:
            question (str): The user's question.
        Returns:
            str: The appropriate DataFrame file name or None if no suitable file is found.
        """
        # 1. Define the response schema fields
        response_schemas = [
            ResponseSchema(name="file_name", description="Name of file which can contain the expected answer", type="string"),
        ]

        # 2. Create the parser
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        # 3. Get format instructions
        format_instructions = output_parser.get_format_instructions()

        prompt_template = """
        
        {files_knowledge_base}

        Your Task is to decide which file could contains the answer to the question and return the file name.

        Question: {question}

        Return your answer in **format** as shown below:
        {format_instructions}

        Other Instructions:
        - If you are not sure, return "None".
        - If you think that the question is not related to any file, return "None".
        - If you think that the question is related to multiple files, return the file name which is most likely to contain the answer.
        - Stick to the response format and do not return any additional information.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = prompt | self.llm | output_parser

        # 7. Call the chain
        result = chain.invoke({"files_knowledge_base": self.files_knowledge_base, "question":question,"format_instructions" : format_instructions})

        print("\n\n Result:", result, "Type:", type(result))

        return result["file_name"] if "file_name" in result and result["file_name"] != "None" else None

    
    def get_answer_from_df(self, state: State) -> State:
        """
        Get answer from the DataFrame based on the user's question.
        Args:
            state (State): The current state of the graph.
        Returns:
            State: The updated state with the answer.
        """

        question = state["messages"][-1].content

        cnt = 3

        file_name = None

        while cnt > 0:

            file_name = self.decide_df_agent(question)
            if file_name is not None and file_name in self.df_agents and file_name != "None":
                break
            cnt-= 1


        if file_name is None or file_name == "None":
            response = {"output": "Despite of multiple attempts, I am not sure which file contains the answer to your question, please try specifying the file name."}
            print(response, type(response))
            print("\n\n\n")
        else:
            state["messages"].append({"role": "assistant", "content": f"Using file: {file_name} to answer your question:{question}."})
            agent = self.df_agents.get(file_name)
            response = agent.invoke({question})
            print(response, type(response))
            print("\n\n\n")

        self.state["messages"].append({"role": "assistant", "content": response["output"]})
        
        return self.state
    
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
        # self.state = {
        #     "messages": [{"role": "user", "content": question}]
        # }
        self.state["messages"].append({"role": "user", "content": question})
        # Run the graph and return final message
        result = self.graph.invoke(self.state)
        return str(result["messages"][-1].content)