"""
Total APIS
1. create session
2. upload file
3. upload text
4. delete file
5. generate_summary
6. qna
7. load data


"""
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
from summ_and_qna import DocSummAndQnA
import tempfile

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
        tmp_path = None
        suffix = file.filename.split('.')[-1]
        # print(suffix, file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            contents = await file.read()  # Await the file read
            tmp.write(contents)
            tmp_path = tmp.name

        data = json.loads(json_data)
        print(data)
        session_id = data["session_id"]

        if session_id not in current_session:
            current_session[session_id] = {}

        print(f"file received, session id created, {session_id}, file path: {tmp_path}")

        # if session_id in current_session:
        
        current_session[str(session_id)]["file_path"] = tmp_path
        current_session[str(session_id)]["input_content"] = "file"

        if tmp_path.endswith(".pdf"):
            current_session[str(session_id)]["file_type"] = "PDF"
        elif tmp_path.endswith(".docs"):
            current_session[str(session_id)]["file_type"] = "DOCS"
        elif tmp_path.endswith(".txt"):
            current_session[str(session_id)]["file_type"] = "TXT"
        # else:
        #     return JSONResponse(status_code=400, content={"error": "session id not "})
        current_session[str(session_id)]["obj"].load_document(file_path = tmp_path, document_type = current_session[str(session_id)]["file_type"])

        return JSONResponse(status_code=200, content={"message": "file received at backend"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"issue at backend during file upload. \nException: {e}"})



@app.get("/create_session")
def create_session():

    try:
        session_id = uuid.uuid4().hex
        current_session[session_id] = defaultdict(dict)
        print(current_session)
        current_session[str(session_id)]["obj"] = DocSummAndQnA()

        return JSONResponse(status_code = 200, content={"session_id": session_id})
    except Exception as e:
            return JSONResponse(status_code=400, content={"error": "issue at backend during session creation"})


if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="localhost", port=8000, reload=True, log_level="debug")