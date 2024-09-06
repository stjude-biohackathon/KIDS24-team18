from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
# from langchain_openai import OpenAI


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

st.session_state["model"] = st.sidebar.selectbox(
    "Select your model?",
    ("llama3.1", "gpt-4o", "gpt-4o-mini", "meditron"),
)

if st.session_state["model"] in ["gpt-4o", "gpt-4o-mini"]:
    st.session_state["api_key"] = st.sidebar.text_input("Paste you OpenAI API key here: ", "sk-xxxxxxx", type = "password")
    st.session_state["base_url"] = "https://api.openai.com/v1"
    # st.session_state["client"] = OpenAI(api_key=password)
else:
    st.session_state["api_key"] = 'ollama'
    st.session_state["base_url"] = "http://localhost:11434/v1"
    # st.session_state["client"] = OpenAI(api_key=st.session_state["password"], base_url = st.session_state["base_url"])

@st.cache_resource
def pandas_agent(uploaded_file):
    df = pd.read_csv(uploaded_file, sep='|', skiprows=2)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)
    df = df[~df.iloc[:, 0].str.contains('-')]

    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model=st.session_state["model"] , api_key= st.session_state["api_key"], base_url=st.session_state["base_url"]),
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
    if "messages" not in st.session_state:
#         messages = [
#     (
#         "system",
#         "You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Your goal is to help people with the QC evaluation for their data and in providing recommendations. Please be as concise as possible in providing your assessment (not extending 300 words). Moreover, please be as critique, as skeptical and as realistic as possible, I want you to be able to provide focus on the low-quality aspects of the data for the human recipient of your message. If you don't find any issues with the data, don't make them up, instead just please write that it all rather looks good etc.,"
#     ),
#     ("human", "Summarize the data"),
# ]
#         response=agent.invoke(messages)
#         st.session_state.messages = [{"role": "assistant", "content": response['output']}]
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the QC analysis?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = agent.invoke(st.session_state.messages)
            bot_output = st.write(response['output'])
        st.session_state.messages.append({"role": "assistant", "content": bot_output})

sidebar_links()
