
from typing import List, Tuple, Dict, Union, Optional, Any
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
import uvicorn
import requests, ast, json
import os
import requests, ast, json
from pydantic import BaseModel
import uuid
import tempfile
from ai_agent import DocQA

from collections import defaultdict

current_session = defaultdict(dict)

# Initialize FastAPI app
app = FastAPI()


@app.delete("/delete_all_session")
def delete_all_session()->JSONResponse:
    """
    Endpoint to delete all active sessions.
    Returns:
        JSONResponse: Response indicating the status of the operation.
    """
    try:
        current_session.clear()
        return JSONResponse(status_code = 200, content={"status": "success", "message": "All sessions reset successfully."})
    except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Issue at backend during resetting all sessions. \nException: {e}"})

@app.post("/delete_session")
def delete_session(json_payload:dict)->JSONResponse:
    """
    Endpoint to delete a specific session based on session_id.
    Args:
        json_payload (dict): JSON payload containing session_id.
    Returns:            
        JSONResponse: Response indicating the status of the operation.
    """

    if "session_id" not in json_payload:
        return JSONResponse(status_code=400, content={"error": "No session id key not provided  "})
    
    if json_payload["session_id"] not in current_session:
        return JSONResponse(status_code=400, content={"error": "Session id not found in active sessions or may be incorrect."})
    
    try:
        session_id = json_payload.get("session_id")
        del current_session[session_id]
        return JSONResponse(status_code = 200, content={"status": "success", "message": f"Session id {session_id} deleted successfully."})
    except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Issue at backend during resetting all sessions. \nException: {e}"})
    

@app.delete("/delete_temp_file")
async def delete_temp_file(json_payload:dict)->JSONResponse:

    if "session_id" not in json_payload:
        return JSONResponse(status_code=400, content={"error": "No session id provided"})
    
    session_id = json_payload.get("session_id")
    if session_id not in current_session:
        print(f"session id not found in current session: {session_id}")
        return JSONResponse(status_code=400, content={"error": f"session id not found in current session: {session_id}"})
    else:
        file_paths = current_session[session_id]["file_paths"]
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"file deleted from path : {file_path}")
            else:
                print(f"file not found at path : {file_path}")

        return JSONResponse(status_code=200, content={"message": "File deleted"})

@app.post("/ask_question")
def ask_question(json_payload: dict)-> JSONResponse:
    """
    Endpoint to ask a question based on the uploaded file.
    Args:
        json_payload (dict): JSON payload containing session_id and question.   
    Returns:
        JSONResponse: Response containing the answer to the question.
    """
    
    try:
        session_id = json_payload.get("session_id")
        question = json_payload.get("question")

        if not session_id or not question:
            return JSONResponse(status_code=400, content={"status": "error","error": "No session id or question provided"})

        if session_id not in current_session:
            return JSONResponse(status_code=400, content={"status": "error","error": "Session id not found in active sessions"})

        response = current_session[session_id]["obj"].ask(question)
        print(f"\n\n answer from docqa: {response}")

        return JSONResponse(status_code=200, content={"status": "success", "answer":response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "error": f"issue at backend during question asking. \nException: {e}"})
    

@app.post("/upload_file")
async def upload_file(files: List[UploadFile] = File(...), json_data: str = Form(...))->JSONResponse:
    """
    Endpoint to upload a file and process it with DocQA.
    Args:
        file (UploadFile): The file to be uploaded.
        json_data (str): JSON string containing session_id and other data.
    Returns:
        JSONResponse: Response containing the status of the file upload and processing.
    """
    # Initialize the response variable
    data = json.loads(json_data)
    if "session_id" not in data:
        return JSONResponse(status_code=400, content={"error": "Key Error: session id key is missing."})
    
    session_id = data["session_id"]
    if session_id not in current_session:
        return JSONResponse(status_code=400, content={"error": "Session doesnot exist or wrong session id provided"})
    if not files:
        return JSONResponse(status_code=400, content={"error": "No file provided, please upload a file"})
    
    response = None
    try:
        preferred_dir = os.path.join(os.getcwd(), "temp")  # Use current working directory
        os.makedirs(preferred_dir, exist_ok=True)  # Ensure directory exists

        file_paths = []
        file_names = []
        file_types = []
        for file in files:
           
            # Extract filename
            file_name = file.filename
            file_path = os.path.join(preferred_dir, file_name)

            file_paths.append(file_path)
            file_names.append(file_name)

            if file_name.endswith(".xlsx"):
                file_types.append("xlsx")
            elif file_name.endswith(".csv"):
                file_types.append("csv")
            else:
                return JSONResponse(status_code=400, content={"error": "Unsupported file type. Only .xlsx and .csv files are allowed."})
            

            # Write file to the preferred directory
            with open(file_path, "wb") as buffer:
                contents = await file.read()
                buffer.write(contents)
            print(f"\n\nFile saved to {file_path}, filename: {file_name}")

        

        print(f"files received, session id created, {session_id}, file paths: {file_paths}")
        
        current_session[str(session_id)]["file_paths"] = file_paths
        current_session[str(session_id)]["file_name"] = file_names
        current_session[str(session_id)]["input_content"] = file_types

        # response = {
        #     "status": "success",
        #     "file_summary": "# All files received at backend Summary currently paused"
        # }
        
        response = current_session[str(session_id)]["obj"].load_document(file_paths = current_session[str(session_id)]["file_paths"], 
                                                                         file_types = current_session[str(session_id)]["input_content"], 
                                                                         file_names = current_session[str(session_id)]["file_name"])
        
        if response["status"] != "success":
            return JSONResponse(status_code=400, content={"error": f"Issue at backend during file load in DocQA Class. \nException: {response['message']}"})
        

        return JSONResponse(status_code=200, content={"message": "file received at backend", "file_summary": response["file_summary"]})
        # return JSONResponse(status_code=200, content={"message": "file received at backend", "file_summary": "# All files received at backend Summary currently paused"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"issue at backend during file upload. \nException: {e}"})


@app.get("/create_session")
def create_session()-> JSONResponse:
    """
    Endpoint to create a new session.
    Returns:
        JSONResponse: Response containing the session ID.
    """

    try:
        session_id = uuid.uuid4().hex
        current_session[session_id] = defaultdict(dict)
        print(current_session)
        current_session[str(session_id)]["obj"] = DocQA()

        return JSONResponse(status_code = 200, content={"session_id": session_id})
    except Exception as e:
            return JSONResponse(status_code=400, content={"error": "issue at backend during session creation"})
    
@app.get("/get_active_sessions")
def get_active_sessions()->JSONResponse:  
    """
    Endpoint to get the list of active sessions.
    Returns:
        JSONResponse: Response containing the list of active session IDs.
    """
    try:
        active_sessions = list(current_session.keys()) if current_session else []
        if not active_sessions:
            return JSONResponse(status_code=200, content={"active_sessions": [], "message": "No active sessions found."})
        return JSONResponse(status_code=200, content={"active_sessions": active_sessions})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Issue at backend during fetching active sessions. \nException: {e}"})

if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="localhost", port=8000, reload=True, log_level="debug")