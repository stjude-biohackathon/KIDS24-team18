from openai import OpenAI
import streamlit as st
from io import StringIO

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

@st.cache_data
def init_summary(uploaded_file):

    # To read file as bytes:
    # bytes_data = '-'.join(StringIO(uploaded_file.getvalue().decode("utf-8")).read())
    uploaded_file = [StringIO(item.getvalue().decode("utf-8")).read() for item in uploaded_file]
    bytes_data = '-'.join(uploaded_file)
    response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that acts as the Computational Biology expert in the area of Epigenetics. Please find the differences and summarize the report to help people with the evaluation for their data. Ignore the 'context.txt file' comment in the report. Please don't hallucinate!"
            },
            {
                "role": "user",
                "content": bytes_data
            }
        ]
            )
    return response.choices[0].message.content

model = st.sidebar.selectbox(
    "Select your model?",
    ("llama3", "gpt-4o", "gpt-4o-mini", "meditron"),
)

if model in ["gpt-4o", "gpt-4o-mini"]:
    password = st.sidebar.text_input("Paste you OpenAI API key here: ", "sk-xxxxxxx", type = "password")
    client = OpenAI(api_key=password)
else:
    client = OpenAI(api_key='ollama', base_url = 'http://localhost:11434/v1')


st.session_state["openai_model"] = model

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files = True)
# st.write(uploaded_file)
if uploaded_file:
    response = init_summary(uploaded_file)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": response}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about the analysis?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
