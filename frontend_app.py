import streamlit as st
import os
import time
import uuid
import json
import requests
from typing import List, Tuple, Dict, Union, Optional, Any

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'welcome_done' not in st.session_state:
    st.session_state.welcome_done = False
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'text_pasted' not in st.session_state:
    st.session_state.text_pasted = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'text_content' not in st.session_state:
    st.session_state.text_content = None

    


# Welcome dialog - only show if not already done
if not st.session_state.welcome_done:
    st.title("Welcome to Document Chat Assistant")
    st.write("This app allows you to upload a document or paste text, then ask questions about it.")

    
    
    # Options for input
    input_method = st.radio("Choose your input method:", ("Upload a file (PDF, TXT, DOCX, XLSX, CSV)", "Paste text directly"))
    
    # File upload or text area based on selection
    if input_method == "Upload a file (PDF, TXT, DOCX, XLSX, CSV)":
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx', 'xlsx', 'csv'])
        if uploaded_file is not None:
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_uploaded = True
    else:
        pasted_text = st.text_area("Paste your text here:", height=200)
        if pasted_text.strip() != "":
            st.session_state.text_content = pasted_text
            st.session_state.text_pasted = True
    
    # API key input
    st.session_state.api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    # Submit button with validation
    if st.button("Submit"):
        if not st.session_state.api_key:
            # st.error("Please enter your OpenAI API key.")
            st.toast('Please enter your OpenAI API key.', icon='⚠️')
        elif not st.session_state.file_uploaded and not st.session_state.text_pasted:
            # st.error("Please either upload a file or paste some text.")
            st.toast('Please either upload a file or paste some text.', icon='⚠️')
        else:
            st.session_state.welcome_done = True
            # Add welcome message from assistant

            st.session_state.messages.append({"role": "assistant", "content": "Welcome! I've processed your document. Here's a sample summary:"})

            with st.spinner("Wait for it...", show_time=True):
                # time.sleep(2)
                
                # create_session_response = requests.get("http://localhost:8000/create_session")
                # content = json.loads(create_session_response.content)
                # st.session_state.session_id = content["session_id"]
                # print(f"\n session id created : {st.session_state.session_id}")

                max_retries = 4
                retry_delay = 1  # seconds
                create_session_response = None

                for attempt in range(1, max_retries + 1):
                    try:
                        create_session_response = requests.get("http://localhost:8000/create_session")
                        content = json.loads(create_session_response.content)
                        
                        session_id = content.get("session_id")
                        
                        if session_id:
                            st.session_state.session_id = session_id
                            print(f"\n✅ Session ID created on attempt {attempt}: {session_id}")
                            break
                        else:
                            print(f"⚠️ Attempt {attempt}: session_id not found, retrying...")

                    except Exception as e:
                        print(f"❌ Attempt {attempt}: Exception occurred - {e}")

                    time.sleep(retry_delay)  # wait before next retry

                if create_session_response.status_code == 200 and st.session_state.session_id is not None:
                    upload_file_response = None
                    upload_text_response = None

                    # print("\n\n\t uploaded file variable \n\t", uploaded_file)

                    if st.session_state.file_uploaded == True:
                        print("\n\n\t inside file upload api \n\n")
                        json_payload = {"session_id":st.session_state.session_id}
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                            "json_data": (None, json.dumps(json_payload), "application/json"),
                            }
                        
                        upload_file_response = requests.post("http://localhost:8000/upload_file", files=files)

                        if upload_file_response.status_code != 200:
                            st.error("Failed to upload file. Please try again.")
                            st.rerun()

                        content = json.loads(upload_file_response.content)

                        print(f"\n\n upload file response content, {content}")

                        st.session_state.summary = content["file_summary"]
                        st.session_state.messages.append({"role": "assistant", "content": st.session_state.summary})
                        st.session_state.messages.append({"role": "assistant", "content": "You can now ask me questions about your document."})
                        # create session backend api
                        print(f"\n\nfile_uploaded, {uploaded_file}")
                    
                    elif st.session_state.text_pasted == True:
                        print("\n\n\t inside abcdefgh \n\n")
                        # json_payload = {
                        #     "session_id": st.session_state.session_id,
                        #     "input_text": st.session_state.text_content
                        #     }
                        # upload_text_response = requests.post("http://localhost:8000/upload_text", json=json_payload)
                        # if upload_text_response.status_code != 200:
                        #     st.error("Failed to upload text. Please try again.")
                        #     st.rerun()
                        # #create session backend api
                        # print("\n\ntext uploaded")
                        pass

                    if st.session_state.file_uploaded == True and upload_file_response.status_code == 200:
                        #delete file
                        delete_file_response = requests.post("http://localhost:8000/delete_temp_file", json={"session_id": st.session_state.session_id})
                        content = json.loads(delete_file_response.content)
                        print(f"\n\nfile deleted, {content}")
            
                    # if st.session_state.session_id is not None:
                    #     # Generate sample summary
                    #     generate_summary_response = requests.post("http://localhost:8000/generate_summary", json={"session_id": st.session_state.session_id})
                    #     content = json.loads(generate_summary_response.content)
                    #     st.session_state.summary = content["file_summary"]
                    #     st.session_state.messages.append({"role": "assistant", "content": st.session_state.summary})
                    #     st.session_state.messages.append({"role": "assistant", "content": "You can now ask me questions about your document."})
                # else:

            
            st.rerun()
    
    st.stop()  # Stop execution here until welcome is done



# Fixed top section
with st.container():
    # Main app after welcome is done
    st.title("Document Chat Assistant")

# if 

st.divider()

# Scrollable chat area
chat_container = st.container()

with chat_container:

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #call qna api
            time.sleep(2)
            json_payload={
                "session_id": st.session_state.session_id,
                "question": prompt
            }
            qna_response = requests.post("http://localhost:8000/ask_question", json=json_payload)
            content = json.loads(qna_response.content)
        st.markdown(content["answer"])
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": content["answer"]})

st.divider()
with st.container():

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.popover("File Info"):
            st.text(f"File Name: {st.session_state.file_name}")
    with col2:
        # Sample data to download
        data = {
            "name": "Document Chat Assistant",
            "version": "1.0",
            "features": ["chat", "document processing", "Q&A"]
        }

        # def download_chat()->dict:
        #     c
        # Convert to JSON string
        json_string = json.dumps(st.session_state.messages, indent=4)

        # Download button
        st.download_button(
            label="📥 Export Chat",
            data=json_string,
            file_name="document_chat_config.json",
            mime="application/json"
        )

    with col3:
        if st.button("Reset"):
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()