from typing import Any
# from .autonotebook import tqdm as notebook_tqdm
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

@st.cache_data
def get_pdf_elements(filebytes: bytes):
    # Get elements
    raw_pdf_elements = partition_pdf(
        file=filebytes,
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=False,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        # Hard max on chunks
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        # image_output_dir_path=path,
    )
    return raw_pdf_elements

class Element(BaseModel):
    type: str
    text: Any

@st.cache_data
def get_element_categories(_raw_pdf_elements: list):
    # Categorize by type
    categorized_elements = []
    for element in _raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    # print(len(table_elements))

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    # print(len(text_elements))
    return table_elements, text_elements

@st.cache_data
def get_summaries(_table_elements: list, _text_elements: list):
    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(temperature=0, model=st.session_state["model"], api_key=st.session_state["api_key"])
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Apply to text
    texts = [i.text for i in _text_elements]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

    # Apply to tables
    tables = [i.text for i in _table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    return text_summaries, table_summaries

@st.cache_resource
def embed_and_retrieve(_texts: list, _tables: list, _text_summaries: list, _table_summaries: list):
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings(openai_api_key=st.session_state["api_key"]))

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in _texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(_text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, _texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in _tables]
    summary_tables = [
        Document(page_content=s, metadata={id_key: table_ids[i]})
        for i, s in enumerate(_table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, _tables)))
    return retriever

def rag_chain(retriever):
    # Prompt template
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Option 1: LLM
    model = ChatOpenAI(temperature=0, model= st.session_state["model"], api_key=st.session_state["api_key"])
    # Option 2: Multi-modal LLM
    # model = GPT4-V or LLaVA

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

@st.cache_resource
def get_retriver(filebytes: bytes):
    raw_pdf_elements = get_pdf_elements(filebytes)
    table_elements, text_elements = get_element_categories(raw_pdf_elements)
    text_summaries, table_summaries = get_summaries(table_elements, text_elements)
    retriever = embed_and_retrieve(text_elements, table_elements, text_summaries, table_summaries)
    return retriever
