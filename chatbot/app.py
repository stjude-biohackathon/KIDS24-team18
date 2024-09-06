import streamlit as st
from openai import OpenAI

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
    ("grumpy","llama3.1", "gpt-4o", "gpt-4o-mini", "meditron"),
)

if st.session_state["model"] in ["gpt-4o", "gpt-4o-mini"]:
    st.session_state["api_key"] = st.sidebar.text_input("Paste you OpenAI API key here: ", "sk-xxxxxxx", type = "password")
    st.session_state["base_url"] = "https://api.openai.com/v1"
    st.session_state["client"] = OpenAI(api_key=t.session_state["api_key"])
else:
    st.session_state["api_key"] = 'ollama'
    st.session_state["base_url"] = "http://localhost:11434/v1"
    st.session_state["client"] = OpenAI(api_key=st.session_state["api_key"], base_url = st.session_state["base_url"])

sidebar_links()

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# def login():
#     if st.button("Log in"):
#         st.session_state.logged_in = True
#         st.rerun()

# def logout():
#     if st.button("Log out"):
#         st.session_state.logged_in = False
#         st.rerun()

# login_page = st.Page(login, title="Log in", icon=":material/login:")
# logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

pathways = st.Page(
    "pathways.py", title="Pathway bot", icon=":material/dashboard:", default=True
)
qc = st.Page("qc.py", title="QC bot", icon=":material/bug_report:")
chatbot = st.Page(
    "chatbot.py", title="chatbot", icon=":material/notification_important:"
)

# search = st.Page("tools/search.py", title="Search", icon=":material/search:")
# history = st.Page("tools/history.py", title="History", icon=":material/history:")

pg = st.navigation(
        {
            "General": [chatbot],
            # "Reports": [pathways, qc],
            "Tools": [pathways, qc],
        }
    )

pg.run()
