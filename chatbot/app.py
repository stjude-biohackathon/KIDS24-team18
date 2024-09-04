from openai import OpenAI
import streamlit as st

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

st.title("Grumpy")
st.write("Grumpy (Generative Research Utility Model in Python) is a tool designed to conduct Biological Context Analysis (BCA). It utilizes Large Language Models (LLMs) such as OpenAI's GPT-4 (St. Jude Dedicated Instance) or other models like Llama from Meta.")

model = st.sidebar.selectbox(
    "Select your model?",
    ("gpt-4o", "gpt-4o-mini", "llama3", "meditron"),
)

if model in ["gpt-4o", "gpt-4o-mini"]:
    password = st.sidebar.text_input("Paste you OpenAI API key here: ", "sk-xxxxxxx", type = "password")
    client = OpenAI(api_key=password)
else:
    client = OpenAI(api_key='ollama', base_url = 'http://localhost:11434/v1')


st.session_state["openai_model"] = model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
