# Document Chat AI Assistant

### Problem Statement

Reading and extracting insights from lengthy or complex documents is time-consuming and inefficient. Users often need to locate specific information or understand content quickly — without going through every page.

### Solution: AI-Powered Document Chat Assistant

An AI agent-enabled web app that allows users to **upload documents or paste text**, and then **chat with an intelligent assistant** to get summaries or ask specific questions about the content — all in real-time.


### Process Flow

1. **User uploads document** (PDF, DOCX, TXT, CSV, etc.) or pastes text.
2. **AI agent processes the content** and generates a summary.
3. User can **chat naturally** with the assistant to ask questions.
4. Responses are generated based on document context.
5. Users can **view file info, export chat, or reset session**.



### Key Features

* Upload multiple file formats
* Paste custom text input
* Chat-based Q\&A with AI agent
* Summary generation
* Export conversations as JSON
* File info viewer & session management

### Benefits

* Saves time by summarizing and querying documents instantly
* Simplifies understanding of dense or unfamiliar content
* Enhances productivity for researchers, professionals, and students
* No need to manually search through files
* Keeps interactions natural and conversational


### Challenges Faced

While building this AI agent-powered document chat assistant, several key challenges were encountered:

1. **Prompt Optimization**
   Crafting effective prompts for consistent and accurate LLM responses was non-trivial. It required iterative testing to guide the model toward producing meaningful and concise answers based on diverse document inputs.

2. **Agent File Routing Logic**
   Ensuring the AI agent could **intelligently decide which file** (among many uploaded) contained the most relevant answer posed a significant challenge. Custom logic and better document context embeddings were required for high accuracy.

3. **Structured LLM Output**
   Raw LLM responses were often too verbose or inconsistent. To ensure reliability, a **Structured Output Parser** was integrated—enforcing a fixed schema for answers, summaries, and metadata.

4. **Frontend-Backend Sync**
   Coordinating the state between **Streamlit (frontend)** and **FastAPI (backend)** was tricky, especially for multi-step workflows (upload → process → answer). Careful management of API calls, session states, and conditional rendering was needed.

5. **API Response Handling**
   Making the user experience smooth required handling errors, edge cases, and retries gracefully. This included meaningful toast messages, retry loops on backend failures, and maintaining context in the UI.


### Under Development & Future Scope

This project is actively evolving to become a more powerful and intelligent document assistant. Here are the upcoming features and improvements:

1. **Multi-Format Input Support**
   Extending full support for PDF, TXT, DOCX, and pasted user text — ensuring seamless ingestion and parsing of various file types, not just tabular ones.

2. **RAG Pipeline for Text Documents**
   Integrating a **Retrieval-Augmented Generation (RAG)** approach to improve accuracy and context understanding for unstructured text files (PDFs, DOCX), which is currently limited to CSV/XLSX.

3. **Ambiguity Handling & Question Classification**
   Enhancing the agent to detect ambiguous or generic user queries and classify whether an answer can be derived from uploaded documents or requires external/contextual knowledge.

4. **Dynamic File Ingestion During Chat**
   Allowing users to **add new documents mid-conversation** without restarting the session — ensuring uninterrupted, evolving conversations.

5. **Advanced Agent Workflow**
   Improving agent decision-making using **LangGraph** and custom logic to enable more nuanced, goal-driven behavior — including task decomposition, memory-aware routing, and reasoning across multiple documents.



``` Note: Some Features are under development and are included in future scope```
---

## Tech Stack

| Layer                  | Technology                                                                        | Purpose                                                                  |
| ---------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Frontend**           | [Streamlit](https://streamlit.io/)                                                | Interactive UI for document upload, chat, and real-time conversation     |
| **Backend**            | [FastAPI](https://fastapi.tiangolo.com/)                                          | High-performance API for session handling, file/text processing          |
| **LLM**                | [OpenAI GPT-3.5-Turbo](https://platform.openai.com/docs)                                  | Natural language understanding and answering document-related queries    |
| **Agentic Framework**  | [LangChain](https://www.langchain.com/) + [LangGraph](https://www.langgraph.dev/) | Building intelligent, looping agents for EDA, Q\&A, and response routing |


---
## Setup Instructions:

### Requirements

* Python 3.10+
* pip
* OpenAI API Key


### Setup Instructions

Follow the steps below to run the project locally:

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/document-chat-assistant.git
cd document-chat-assistant
```

#### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set OpenAI API Key

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your-openai-api-key
```

> 🔐 Make sure not to commit `.env` to source control.

#### 5. Run Backend (FastAPI)

In a new terminal (from project root):

```bash
python3 backend_api.py
```

#### 6. Run Frontend (Streamlit)

In a separate terminal:

```bash
streamlit run backend_api.py
```


--

## Repository Files and Code Structure


### File Name: `summary_generator.py`

This Python module defines an abstract interface and concrete implementation for generating exploratory data analysis (EDA) summaries and questions from a given pandas DataFrame. It utilizes the LangChain framework and a language model (LLM, e.g., OpenAI) to interpret the dataset and generate insights.

**Key functionality includes:**

* Displaying basic metadata about the DataFrame (`info`, `head`, column info).
* Automatically generating a summary and top 10 EDA questions using an LLM.
* Abstract class design for flexibility and future extension.


### File Code Structure

#### 1. **Abstract Base Class: `GenerateFileSummary`**

Defines the interface for generating summaries and EDA questions.

#### 🔹 `__init__(self, llm=None)`

* **Description**: Initializes the base class with an optional language model (`llm`) instance.
* **Arguments**:

  * `llm` (optional): A language model to be used for generation.

##### `generate_summary(self, df, file_name, file_format)`

* **Description**: Abstract method for generating a summary and EDA questions from a DataFrame.
* **Returns**: Tuple containing:

  * Markdown summary string,
  * List of EDA questions,
  * Full markdown-formatted summary.



#### 2. **Concrete Class: `DataFrameSummaryGenerator`**

Implementation of the abstract class that generates a structured summary and EDA questions from a DataFrame using LangChain and an LLM.

##### `__init__(self, llm=None)`

* **Description**: Initializes the summary generator and optionally assigns a language model.
* **Arguments**:

  * `llm`: The language model instance to be used in the pipeline.

##### `dataframe_basic_info(df: pd.DataFrame) -> str` (Static Method)

* **Description**: Produces a basic metadata summary of the provided DataFrame including:

  * Output of `df.info()`,
  * First 5 rows using `df.head()`,
  * List of column names,
  * Top 10 unique values in all object-type columns.
* **Returns**: A formatted string containing all the above information.

##### `generate_summary(self, df, file_name=None, file_format=None) -> Tuple[str, List[str], str]`

* **Description**:

  * Generates a high-level summary and a list of 10 EDA questions using the provided LLM.
  * Combines the results into a single markdown output.
* **Steps Involved**:

  1. Extract basic information from the DataFrame.
  2. Define response schema for structured output (summary + questions).
  3. Build LangChain prompt and processing chain.
  4. Invoke the chain with formatted instructions.
  5. Parse and return the results.
* **Returns**: Tuple of:

  * DataFrame summary (str),
  * List of 10 EDA questions (List\[str]),
  * Markdown-formatted summary (str).


### Usage Example

```python
from summary_generator import DataFrameSummaryGenerator
import pandas as pd

df = pd.read_csv("sample.csv")
generator = DataFrameSummaryGenerator(llm=my_llm)
summary, questions, markdown = generator.generate_summary(df, file_name="sample.csv", file_format="csv")
print(markdown)
```
---


### File Name: `ai_agent.py`

The `ai_agent.py` module provides a framework for intelligent document-based Question Answering (QA) using language models and structured document understanding. It leverages LangChain’s agent and graph capabilities along with pandas DataFrames to generate summaries and handle natural language queries over data files (CSV/XLSX).

Uses **LangChain**, **LangGraph**, **Pandas**, and **OpenAI's GPT** to build a conversational agent capable of:

  * Loading CSV/XLSX files as pandas DataFrames
  * Summarizing content with domain and EDA questions
  * Choosing the relevant file for a question
  * Executing the query over the appropriate DataFrame agent
  * Structuring the entire flow using a **state graph**

### File Code Structure

#### Class `State`

* **Type**: `TypedDict`
* **Description**: Defines the structure of the graph state with a message list used by LangGraph.


#### Class `DocQA`

> Main orchestrator class for document-based question answering.

##### **`__init__(self)`**

* Initializes the DocQA agent.
* Sets up:

  * OpenAI LLM
  * Empty dictionaries for files, summaries, agents, and DataFrames
  * Initial messages and knowledge base strings
  * Compiles the state graph

##### **`load_document(self, file_paths, file_types, file_names) -> Dict[str, Any]`**

* Loads CSV/XLSX files and creates:

  * Pandas DataFrames
  * Agents for querying the DataFrame
  * File summaries and EDA questions
* Stores structured info in memory
* Returns a status dictionary

##### **`decide_df_agent(self, question: str) -> Optional[str]`**

* Uses the LLM to select the most relevant file (by name) that likely contains the answer to the user’s question.
* Returns the file name or `None` if no match is found.


##### **`get_answer_from_df(self, state: State) -> State`**

* Retrieves the user question from the message state.
* Determines the best matching file using `decide_df_agent`.
* Queries the corresponding DataFrame agent.
* Appends the assistant's answer to the conversation history.
* Returns the updated state.


##### **`_build_graph(self) -> StateGraph`**

* Builds a LangGraph `StateGraph` for processing document queries.
* The graph has one node:

  * `get_answer_from_df` (which loops through answering)
* Returns the compiled graph.


##### **`ask(self, question: str) -> Union[str, dict]`**

* Public method to submit a question to the agent.
* Updates the current message state with the user's question.
* Invokes the state graph and returns the assistant's response.



### Example Flow

```python
docqa = DocQA()
docqa.load_document(["sales_data.csv"], ["csv"], ["Sales Report"])
response = docqa.ask("What was the top-selling product in Q1?")
print(response)
```

### Notes

* PDF, DOCS, TXT file support is stubbed and not yet implemented.
* Supports only `.csv` and `.xlsx` files currently.
* The agent retries 3 times to find the relevant file before giving up.
* Need to make agent workflow more efficient and intellignent.

---

### File Name `backend_api.py`:

`backend_api.py` is a FastAPI-based backend service that provides endpoints to manage document processing sessions, upload and analyze files, and interact with a document-based QA system (`DocQA`). It supports session creation, question answering, file uploads, and session lifecycle management using temporary file storage.



#### File Contents Description

- **Backend Framework**: FastAPI
- **Core Functionality**:
  - Session management
  - File upload and type validation (.csv, .xlsx)
  - Integration with a custom `DocQA` agent for question-answering
  - Temporary file management
- **Storage**: Uses a Python `defaultdict` to hold session metadata in memory

##### API Endpoints

#### `GET /create_session`

* **Description**: Creates a new session and initializes a `DocQA` instance.
* **Returns**: `session_id` (UUID) in JSON.

#### `GET /get_active_sessions`

* **Description**: Lists all currently active session IDs.
* **Returns**: List of session IDs or a message if none are active.

#### `DELETE /delete_all_session`

* **Description**: Deletes all active sessions by clearing the `current_session` dictionary.
* **Returns**: Success or error message.

#### `POST /delete_session`

* **Description**: Deletes a specific session by `session_id`.
* **Arguments**:

  * `json_payload`: `{ "session_id": str }`
* **Returns**: Success or error message.

#### `DELETE /delete_temp_file`

* **Description**: Deletes all temporary files associated with a session.
* **Arguments**:

  * `json_payload`: `{ "session_id": str }`
* **Returns**: File deletion message.

#### `POST /upload_file`

* **Description**: Uploads and processes files (CSV/XLSX) for a given session.

* **Arguments**:

  * `files`: List of `UploadFile`
  * `json_data`: Form field with session metadata (`session_id`)

* **Returns**: File processing summary or error.

* **Steps**:

  * Validates file extensions
  * Saves files to a local `temp/` directory
  * Updates `current_session` with metadata
  * Loads the document in the `DocQA` agent

#### `POST /ask_question`

* **Description**: Asks a question based on uploaded files in a session.
* **Arguments**:

  * `json_payload`: `{ "session_id": str, "question": str }`
* **Returns**: Answer from the `DocQA` model

### 📡 API Endpoints

| API Name               | Method | Description                                                      | Arguments                                      |
|------------------------|--------|------------------------------------------------------------------|------------------------------------------------|
| `/create_session`      | GET    | Creates a new session and initializes a `DocQA` instance         | None                                           |
| `/get_active_sessions` | GET    | Lists all currently active session IDs                           | None                                           |
| `/delete_all_session`  | DELETE | Deletes all active sessions by clearing session state            | None                                           |
| `/delete_session`      | POST   | Deletes a specific session by `session_id`                       | `{"session_id": str}`                          |
| `/delete_temp_file`    | DELETE | Deletes temporary files for a given session                      | `{"session_id": str}`                          |
| `/upload_file`         | POST   | Uploads and processes files (CSV/XLSX) for a given session       | `files`, `json_data={"session_id": str}`       |
| `/ask_question`        | POST   | Asks a question based on uploaded files in a session             | `{"session_id": str, "question": str}`         |


### Run the App Locally

```bash
python3 backend_api.py
```
---
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### File Name: `frontend_app.py`

`frontend_app.py` is a **Streamlit-based Document Chat Assistant** that allows users to upload documents or paste text, process the content via backend APIs, and interact with the document in a conversational chat interface.

This application integrates with a backend service to:

* Create a session
* Upload and process documents or raw text
* Generate a summary
* Handle question-answering based on the document content
* Export chat history
* View uploaded file info
* Reset the session

### Features

* **Document Upload Support:** Accepts PDF, TXT, DOCX, XLSX, CSV formats
* **Paste Raw Text:** Alternative to uploading a file
* **OpenAI API Key Input:** Required for backend interactions
* **Chat Interface:** Ask contextual questions about your document
* **Session Management:** Manages session lifecycle using a backend service
* **Export Chat History:** Save conversation as a JSON file
* **Uploaded File Info:** View file names and types
* **Reset Button:** Clear session and restart interaction

### Code Flow & Logic

#### 1. **Session Initialization**

Initializes session variables:

```python
st.session_state['messages'] = []
st.session_state['api_key'] = ""
...
```

#### 2. **Welcome Screen**

Displayed only once per session:

* Prompt for file upload or text input
* Input for OpenAI API key
* Validates inputs before proceeding

#### 3. **Backend Session Creation**

Attempts to create a session by calling:

```http
GET /create_session
```

If successful, stores the session ID.

#### 4. **File/Text Upload to Backend**

* For files:

  ```http
  POST /upload_file
  ```
* For pasted text (not functional currently):

  ```http
  POST /upload_text
  ```

Once uploaded, the app displays a summary and allows the user to ask questions.

#### 5. **Q\&A Chatbot Interface**

* User inputs a question
* Backend is called with:

  ```http
  POST /ask_question
  ```
* Answer is rendered in chat and saved to session

#### 6. **Chat Export**

Button to download the entire interaction (including file info, date, and messages) as a JSON file.

#### 7. **File Info Dialog**

Popup showing uploaded file names and MIME types.

#### 8. **Session Reset**

Button calls:

```http
POST /delete_session
```

Clears all session variables and restarts the app.

### How to Run

```bash
streamlit run frontend_app.py
```

### Example Usage

1. Upload a `.pdf` or paste text
2. Enter your OpenAI API key
3. Click **Submit**
4. View a summary and start chatting
5. Export your chat or reset session as needed

## Notes

* Ensure the backend service (`localhost:8000`) is up and running before launching the app.
* All interactions are stored in `st.session_state` to maintain chat continuity.

