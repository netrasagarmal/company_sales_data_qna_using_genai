import os, time, uuid, json, requests
from typing import List, Tuple, Dict, Union, Optional, Any
from datetime import date
from collections import defaultdict
import streamlit as st

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

if 'file_names' not in st.session_state:
    st.session_state.file_names = defaultdict(dict)

if 'text_content' not in st.session_state:
    st.session_state.text_content = None

# Welcome dialog - only show if not already done
if not st.session_state.welcome_done:
    st.title("Welcome to Document Chat Assistant")
    st.write("This app allows you to upload a document or paste text, then ask questions about it.")

    
    uploaded_files:  = None
    pasted_text = None

    # Options for input
    input_method = st.radio("Choose your input method:", ("Upload a file (PDF, TXT, DOCX, XLSX, CSV)", "Paste text directly"))
    
    # File upload or text area based on selection
    if input_method == "Upload a file (PDF, TXT, DOCX, XLSX, CSV)":
        uploaded_files = st.file_uploader("Choose a file(s)", type=['pdf', 'txt', 'docx', 'xlsx', 'csv'], accept_multiple_files=True)
        if uploaded_files is not None:
            # st.session_state.file_name = uploaded_file.name
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

            st.session_state.messages.append({"role": "assistant", "content": "Welcome! I've processed your document. Here's a sample summary:"})

            with st.spinner("Wait for it...", show_time=True):
                
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

                # Check if session ID was created successfully
                if create_session_response.status_code == 200 and st.session_state.session_id is not None:
                    
                    upload_file_response = None
                    upload_text_response = None

                    # If a file is uploaded or text is pasted, send it to the backend
                    if st.session_state.file_uploaded == True:
                        json_payload = {"session_id":st.session_state.session_id}
                    
                        files = []

                        # Add all files under the field name 'files'
                        for file in uploaded_files:
                            files.append(
                                ("files", (file.name, file.getvalue(), file.type))
                            )
                            st.session_state.file_names[file.name] = file.type

                        # Add JSON payload as a form field
                        files.append(
                            ("json_data", (None, json.dumps(json_payload), "application/json"))
                        )

                        upload_file_response = requests.post("http://localhost:8000/upload_file", files=files)

                        if upload_file_response.status_code != 200:
                            st.error("Failed to upload file. Please try again.")
                            st.rerun()

                        content = json.loads(upload_file_response.content)

                        st.session_state.summary = content["file_summary"]
                        st.session_state.messages.append({"role": "assistant", "content": st.session_state.summary})
                        st.session_state.messages.append({"role": "assistant", "content": "You can now ask me questions about your document."})
                        # create session backend api
                    
                    elif st.session_state.text_pasted == True:
                        print("\n\n\t inside abcdefgh \n\n")
                        json_payload = {
                            "session_id": st.session_state.session_id,
                            "input_text": st.session_state.text_content
                            }
                        upload_text_response = requests.post("http://localhost:8000/upload_text", json=json_payload)
                        if upload_text_response.status_code != 200:
                            st.error("Failed to upload text. Please try again.")
                            st.rerun()
                        #create session backend api
                        

                    if st.session_state.file_uploaded == True and upload_file_response.status_code == 200:
                        #delete file
                        delete_file_response = requests.delete("http://localhost:8000/delete_temp_file", json={"session_id": st.session_state.session_id})
                        content = json.loads(delete_file_response.content)
            
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
        # with st.popover("File Info"):
        #     st.text(f"File Name: {st.session_state.file_name}")
        # Dialog to show file info
        @st.dialog("Uploaded File Info")
        def show_file_info():
            if st.session_state.file_names:
                for file_name, file_type in st.session_state.file_names.items():
                    st.write(f"📄 **{file_name}** — *{file_type}*")
            else:
                st.info("No files uploaded yet.")

        # Button to trigger dialog
        if st.button("Show File Info"):
            show_file_info()

    with col2:

        download_data = {
            "date":date.today().strftime("%Y-%m-%d"),
            "files": st.session_state.file_names,
            "conversation": st.session_state.messages,
            "api_key": st.session_state.api_key
        }
        
        json_string = json.dumps(download_data, indent=4)

        # Download button
        st.download_button(
            label="📥 Export Chat",
            data=json_string,
            file_name="document_chat_config.json",
            mime="application/json"
        )

    with col3:
        if st.button("Reset"):
            delete_current_session_response = requests.post("http://localhost:8000/delete_session", json={"session_id": st.session_state.session_id})
            if delete_current_session_response.status_code == 200:
                print("Session deleted successfully.")
                # Clear all session state variables
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            else:
                print("Failed to delete session.")
                st.error("Failed to delete session. Please try again.")