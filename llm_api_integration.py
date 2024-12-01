import streamlit as st
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI

path_file = "C:/Users/USER\Rusnandi Fikri/master_keys.xlsx"
API_KEY = pd.read_excel(path_file)["api_key"][1]

def chat(question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        api_key=API_KEY
        )
    messages = [
        (
            "system",
            "You are a helpful assistant. Always answer in Indonesian language.",
        ),
        ("human", question),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg

st.title("QnA Gemini Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat(prompt)
    answer = response.content

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})