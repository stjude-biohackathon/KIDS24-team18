from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
# from langchain_openai import OpenAI


@st.cache_resource
def pandas_agent(uploaded_file):
    df = pd.read_csv(uploaded_file, sep='|', skiprows=2)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df = df[~df.iloc[:, 0].str.contains('-')]

    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=""),
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type=AgentType.OPENAI_FUNCTIONS)
    json_llm = agent.bind(response_format={"type": "json_object"})

    return json_llm

uploaded_file = st.file_uploader("Upload a QC file") #, accept_multiple_files = True)
# st.write(uploaded_file)
if uploaded_file:
    agent = pandas_agent(uploaded_file)
    # if "messages" not in st.session_state:
    messages = [
    (
        "system",
        "You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the QC evaluation for their data and in providing recommendations. Please be as concise as possible in providing your assessment (not extending 300 words). Moreover, please be as critique, as skeptical and as realistic as possible, I want you to be able to provide focus on the low-quality aspects of the data for the human recipient of your message. If you don't find any issues with the data, don't make them up, instead just please write that it all rather looks good etc.,"
    ),
    ("human", "Summarize the data"),
]
    response=agent.invoke(messages)
    st.session_state.messages.append([{"role": "assistant", "content": response['output']}])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the QC analysis?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = agent.invoke(st.session_state.messages)
            response = st.write(response['output'])
        st.session_state.messages.append({"role": "assistant", "content": response})
