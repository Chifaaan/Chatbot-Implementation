import streamlit as st
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2


def chat(contexts, history, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=API_KEY
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You can use given data to answer question about product.",
            ),
            ("human", "This is the data : {contexts}\nUse this chat history to generate relevant answer from recent conversation: {history}\nUser question : {question}"),
        ]
    )
    
    chain = prompt | llm
    completion = chain.invoke(
        {
            "contexts": contexts,
            "history": history,
            "question": question,
        }
    )

    answer = completion.content
    # input_tokens = completion.usage_metadata['input_tokens']
    # completion_tokens = completion.usage_metadata['output_tokens']

    result = {}
    result["answer"] = answer
    # result["input_tokens"] = input_tokens
    # result["completion_tokens"] = completion_tokens
    return result

st.title("AI Chatbot Assistant")

# Sidebar for API Key Validation and File Upload
with st.sidebar:
    st.sidebar.title("Configuration")
    api_valid = False

    # API Key Input
    API_KEY = st.text_input("Enter your Gemini API Key", type="password")
    if API_KEY:
        try:
            # Validate API key
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                api_key=API_KEY
            )
            llm.predict("Hello")
            st.success("API key is valid.")
            api_valid = True
        except Exception as e:
            st.error("Invalid API key. Please check and try again.")
    else:
        st.info("Enter your API key to proceed.")

# Show file uploader only if API key is valid
file_exist = False
if api_valid:
    with st.sidebar:
        uploaded_files = st.sidebar.file_uploader(
            "Choose one or more files",
            type=["csv", "xls", "xlsx", "pdf"],
            accept_multiple_files=True
        )

        contexts = ""  # Initialize contexts

        if uploaded_files:
            try:
                all_contexts = []  # To store content from all files

                for uploaded_file in uploaded_files:
                    # Check file extension and process accordingly
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        df = df.drop_duplicates()
                        all_contexts.append(df.to_string())
                    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(uploaded_file)
                        df = df.drop_duplicates()
                        all_contexts.append(df.to_string())
                    elif uploaded_file.name.endswith('.pdf'):
                        # Extract text from PDF
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text()
                        all_contexts.append(pdf_text.strip())  # Append extracted text
                    else:
                        st.sidebar.error(f"Unsupported file format: {uploaded_file.name}")
                        continue

                # Display preview for each file
                for file_content, file in zip(all_contexts, uploaded_files):
                    st.sidebar.success(f"File `{file.name}` has been successfully processed!")

                # Combine all file contents into a single context string
                contexts = "\n\n".join(all_contexts)
                file_exist = True

            except Exception as e:
                st.sidebar.error(f"An error occurred while processing the files: {e}")
        else:
            st.sidebar.info("No files uploaded yet.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add dynamic assistant messages based on state
notif = st.chat_message("assistant")
if not api_valid:
    notif.write("Enter your API key to proceed.")
elif not file_exist:
    notif.write("Upload your files to proceed.")
else:
    notif.write(f"My knowledge has been updated with {len(uploaded_files)} files. Ask me questions!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ensure `history` is always defined
if st.session_state.messages:
    messages_history = st.session_state.messages[-10:]  # Get the last 10 messages
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history])
else:
    history = ""  # Initialize empty history if no messages are available

# React to user input
if file_exist:
    if prompt := st.chat_input("What is up?"):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        with st.spinner("Assistant is thinking..."):
            response = chat(contexts, history, prompt)
            answer = response["answer"]
        # input_tokens = response["input_tokens"]
        # completion_tokens = response["completion_tokens"]

        # # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     st.markdown(answer)
        #     container = st.container(border=True)
        #     container.write(f"Input Tokens : {input_tokens}")
        #     container.write(f"Completion Tokens: {completion_tokens}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
