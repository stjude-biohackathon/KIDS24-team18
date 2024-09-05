from openai import OpenAI
import streamlit as st
from io import StringIO

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Please give me some context about the analysis! For example, \n1. Data type \n2. Data source \n3. Data analysis goal \n4. Data analysis method, etc."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the analysis?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state["client"].chat.completions.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


