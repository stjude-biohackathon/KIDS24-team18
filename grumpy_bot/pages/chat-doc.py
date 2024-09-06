# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
# import pandas as pd
# from langchain_openai import OpenAI
import vectorstore as vs


st.set_page_config(
    page_title="Grumpy",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/stjude-biohackathon/KIDS24-team18/issues",
        'About': "Grumpy (Generative Research Utility Model in Python) is a tool designed to conduct Biological Context Analysis (BCA). It utilizes Large Language Models (LLMs) such as OpenAI's GPT-4 (St. Jude Dedicated Instance) or other models like Llama from Meta."
    }
)

# col1, col2 = st.columns([1, 2])
st.sidebar.image("templates/Grumpy_logo_250x250.png", caption="Grumpy")
st.title("Grumpy")
st.markdown("Hi :wave: I'm Grumpy (Generative Research Utility Model in Python). I'm designed to conduct Biological Context Analysis (BCA) utilizing Large Language Models (LLMs) such as OpenAI's GPT-4o-mini or other open-source models like Llama3.1 from Meta.")


@st.cache_data
def sidebar_links():

    software_link_dict = {
        "GitHub Page": "https://github.com/stjude-biohackathon/KIDS24-team18",
        "Ollama": "https://ollama.com/",
        "Streamlit": "https://streamlit.io",
    }

    st.sidebar.markdown("## Software-Related Links")
    link_1_col, link_2_col, link_3_col = st.sidebar.columns(3)
    i = 0
    link_col_dict = {0: link_1_col, 1: link_2_col, 2: link_3_col}
    for link_text, link_url in software_link_dict.items():
        st_col = link_col_dict[i]
        i += 1
        if i == len(link_col_dict.keys()):
            i = 0
        st_col.markdown(f"[{link_text}]({link_url})")

    st.sidebar.markdown("## Contact Us")
    contact_link_dict = {
        "Wojciech Rosikiewicz":"https://github.com/forrest1988",
        "Tarun Mamidi": "https://github.com/tkmamidi",
        "Shaurita Hutchins": "https://github.com/sdhutchins",
        "Wenjie Qi": "https://github.com/WenjieQi",
        "Felicia Iordachi":"https://github.com/felicia-19",
        "Farzaan Quadri":""
    }

    # st.sidebar.markdown("## Contact-Related Links")
    for link_text, link_url in contact_link_dict.items():
        st.sidebar.markdown(f"[{link_text}]({link_url})")

uploaded_file = st.file_uploader("Upload a pdf report") #, accept_multiple_files = True)
# st.write(uploaded_file)
if uploaded_file:
    # agent = pandas_agent(uploaded_file)
    retriver = vs.get_retriver(uploaded_file)
    custom_agent = vs.rag_chain(retriver)
    # st.write(raw_pdf_elements[0])
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the analysis?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = custom_agent.invoke(prompt)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

sidebar_links()
