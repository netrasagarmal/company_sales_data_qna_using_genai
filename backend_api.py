
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
async def upload_file(file: UploadFile = File(...), json_data: str = Form(...)):
    try:
        # Custom directory where you want to save the file
        # preferred_dir = "G:/company_sales_genai/company_sales_data_qna_using_genai/temp"
        preferred_dir = os.path.join(os.getcwd(), "temp")  # Use current working directory
        os.makedirs(preferred_dir, exist_ok=True)  # Ensure directory exists

        # Extract filename
        filename = file.filename
        file_path = os.path.join(preferred_dir, filename)

        # Write file to the preferred directory
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # Separate file path and filename
        file_name = os.path.basename(file_path)
        full_file_path = os.path.abspath(file_path)

        print(f"\n\nFile saved to {full_file_path}, filename: {file_name}")

        data = json.loads(json_data)
        print(data)
        session_id = data["session_id"]

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
        # else:
        #     return JSONResponse(status_code=400, content={"error": "session id not "})
        response = current_session[str(session_id)]["obj"].load_document(file_path = current_session[str(session_id)]["file_path"], 
                                                              file_type=current_session[str(session_id)]["file_type"], 
                                                              file_name='forcast.xlsx')
        
        if response["status"] == "exception":
            return JSONResponse(status_code=400, content={"error": f"Issue at backend during file load in DocQA Class. \nException: {response['message']}"})
        
        # ans = current_session[str(session_id)]["obj"].ask("give all the object columns in the dataframe")
        # print(f"\n\n answer from docqa: {ans}")

        return JSONResponse(status_code=200, content={"message": "file received at backend"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"issue at backend during file upload. \nException: {e}"})

@app.post("/ask_question")
def ask_question(json_payload: dict):
    try:
        session_id = json_payload.get("session_id")
        question = json_payload.get("question")

        if not session_id or not question:
            return JSONResponse(status_code=400, content={"error": "No session id or question provided"})

        if session_id not in current_session:
            return JSONResponse(status_code=400, content={"error": "Session id not found in active sessions"})
        


        response = current_session[session_id]["obj"].ask(question)
        print(f"\n\n answer from docqa: {response}")

        return JSONResponse(status_code=200, content={"message": response})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"issue at backend during question asking. \nException: {e}"})


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


if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="localhost", port=8000, reload=True, log_level="debug")