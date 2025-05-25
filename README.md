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


```Note: Some Features are under development and are included in future scope```
---
Here’s a clean and concise **Tech Stack** section for your README, structured for professional presentation and clarity:

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

## Repository files and code structure