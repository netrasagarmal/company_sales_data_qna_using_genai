
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
    
@app.post("/delete_temp_file")
async def delete_temp_file(json_payload:dict):
    
    session_id = json_payload.get("session_id")
    if not session_id:
        return JSONResponse(status_code=400, content={"error": "No session id provided"})
    else:
        file_path = current_session[session_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"file deleted from path : {file_path}")

    return JSONResponse(status_code=200, content={"message": "File deleted"})


@app.post("/upload_file")
async def upload_file(files: List[UploadFile] = File(...), json_data: str = Form(...))->JSONResponse :
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
    session_id = data["session_id"]
    if not session_id:
        return JSONResponse(status_code=400, content={"error": "No session id provided"})
    if not file:
        return JSONResponse(status_code=400, content={"error": "No file provided"})
    
    if session_id not in current_session:
        current_session[session_id] = {}
    

    response = None
    try:
        preferred_dir = os.path.join(os.getcwd(), "temp")  # Use current working directory
        os.makedirs(preferred_dir, exist_ok=True)  # Ensure directory exists

        file_paths = []
        file_names = []
        file_types = []
        for file in files:
            # contents = await file.read()
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

        # if session_id in current_session:
        
        current_session[str(session_id)]["file_paths"] = file_paths
        current_session[str(session_id)]["file_name"] = file_names
        current_session[str(session_id)]["input_content"] = file_types

        
        # else:
        #     return JSONResponse(status_code=400, content={"error": "session id not "})
        response = current_session[str(session_id)]["obj"].load_document(file_path = current_session[str(session_id)]["file_path"], 
                                                              file_type=current_session[str(session_id)]["file_type"], 
                                                              file_name='forcast.xlsx')
        
        if response["status"] != "success":
            return JSONResponse(status_code=400, content={"error": f"Issue at backend during file load in DocQA Class. \nException: {response['message']}"})
        

        return JSONResponse(status_code=200, content={"message": "file received at backend", "file_summary": response["file_summary"]})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"issue at backend during file upload. \nException: {e}"})
    
    # return JSONResponse(status_code=200, content={"message": "file received at backend", "file_summary": response["file_summary"]})


@app.get("/create_session")
def create_session():

    try:
        session_id = uuid.uuid4().hex
        current_session[session_id] = defaultdict(dict)
        print(current_session)
        current_session[str(session_id)]["obj"] = DocQA()

        return JSONResponse(status_code = 200, content={"session_id": session_id})
    except Exception as e:
            return JSONResponse(status_code=400, content={"error": "issue at backend during session creation"})
    

@app.post("/upload_multiple_files/")
async def upload_files(files: List[UploadFile] = File(...)):
    preferred_dir = os.path.join(os.getcwd(), "temp")  # Use current working directory
    os.makedirs(preferred_dir, exist_ok=True)  # Ensure directory exists

    file_paths = []
    file_names = []
    file_types = []
    for file in files:
        # contents = await file.read()
        # Extract filename
        file_name = file.filename
        file_path = os.path.join(preferred_dir, file_name)

        file_paths.append(file_path)
        file_names.append(file_name)

        # Write file to the preferred directory
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        print(f"\n\nFile saved to {file_path}, filename: {file_name}")

        if session_id not in current_session:
            current_session[session_id] = {}

        print(f"file received, session id created, {session_id}, file path: {full_file_path}")

        # if session_id in current_session:
        
        current_session[str(session_id)]["file_path"] = full_file_path
        current_session[str(session_id)]["file_name"] = file_name
        current_session[str(session_id)]["input_content"] = "file"

        if file_name.endswith(".xlsx"):
            current_session[str(session_id)]["file_type"] = "xlsx"
        elif file_name.endswith(".csv"):
            current_session[str(session_id)]["file_type"] = "csv"
        
    return {"files": results}



if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="localhost", port=8000, reload=True, log_level="debug")