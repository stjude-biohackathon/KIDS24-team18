from openai import OpenAI
import streamlit as st
from io import StringIO

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


if "messages" not in st.session_state:
    st.session_state.messages = []

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


