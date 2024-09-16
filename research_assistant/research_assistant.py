import os
import sys
import logging
import yaml
import streamlit as st
from dotenv import load_dotenv
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from enterprise_knowledge_retriever.src.document_retrieval import DocumentRetrieval
from enterprise_knowledge_retriever.src.langgraph_rag import RAG
from search_assistant.src.search_assistant import SearchAssistant
from utils.model_wrappers.api_gateway import APIGateway
from utils.vectordb.vector_db import VectorDb

CONFIG_PATH = os.path.join(current_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, "data/my-vector-db")

load_dotenv(os.path.join(repo_dir, '.env'))

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")

def load_config():
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def init_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True
    if "search_assistant" not in st.session_state:
        st.session_state.search_assistant = None

def handle_document_upload(documentRetrieval):
    docs = st.file_uploader(
        "Add files", accept_multiple_files=True,
        type=[".pdf",".doc",".docx"]
    )
    if st.button("Process"):
        with st.spinner("Processing"):
            text_chunks = documentRetrieval.parse_doc(docs)
            embeddings = documentRetrieval.load_embedding_model()
            st.session_state.embeddings = embeddings
            vectorstore = documentRetrieval.create_vector_store(text_chunks, embeddings, output_db=None)
            st.session_state.vectorstore = vectorstore

            # Initialize or update SearchAssistant for Web Search mode
            if 'search_assistant' not in st.session_state or st.session_state.search_assistant is None:
                config = load_config()
                st.session_state.search_assistant = SearchAssistant(config, vectorstore)
            else:
                st.session_state.search_assistant.vectorstore = vectorstore

            # Initialize RAG for Document Chat mode
            rag = RAG(
                config_path=CONFIG_PATH,
                embeddings=st.session_state.embeddings,
                vectorstore=st.session_state.vectorstore
            )
            lg_configs = rag.load_config(CONFIG_PATH)
            rag.init_llm()
            rag.init_qa_chain()
            rag.init_final_generation()
            workflow = rag.create_rag_nodes()
            rag.build_rag_graph(workflow)
            st.session_state.conversation = rag

            st.toast("File uploaded! Go ahead and ask some questions", icon='🎉')
            st.session_state.input_disabled = False


def handle_web_search():
    if st.session_state.search_assistant is None:
        config = load_config()
        st.session_state.search_assistant = SearchAssistant(config)
    
    query = st.text_input("Enter your search query")
    if query:
        with st.spinner("Searching..."):
            result = st.session_state.search_assistant.basic_call(query)
            st.write(result['answer'])
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    for source in result['sources']:
                        st.write(source)

def handle_user_input(user_question, mode):
    if user_question:
        with st.spinner("Processing..."):
            if mode == "Document Chat":
                response = st.session_state.conversation.call_rag(user_question)
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(response["answer"])
                
                sources = set([f'{sd.metadata["filename"]}' for sd in response["source_documents"]])
                sources_text = ""
                for index, source in enumerate(sources, start=1):
                    sources_text += f'<font size="2" color="grey">{index}. {source}</font> \n'
                st.session_state.sources_history.append(sources_text)
            else:
                if st.session_state.search_assistant is None:
                    config = load_config()
                    st.session_state.search_assistant = SearchAssistant(config)
                
                response = st.session_state.search_assistant.basic_call(user_question)
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(response["answer"])
                
                sources_text = ""
                for index, source in enumerate(response["sources"], start=1):
                    sources_text += f'<font size="2" color="grey">{index}. {source}</font> \n'
                st.session_state.sources_history.append(sources_text)

    for ques, ans, source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
    ):
        with st.chat_message("user"):
            st.write(f"{ques}")
        
        with st.chat_message("ai", avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png"):
            st.write(f"{ans}")
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    st.markdown(f'<font size="2" color="grey">{source}</font>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="/home/giri/ai-starter-kit/images/research_ai-png.png",
    )

   
    col1, col2 = st.columns([1, 4])


    with col1:
        logo = Image.open("/home/giri/ai-starter-kit/images/research_ai-png.png")
        st.image(logo, width=100) 

    with col2:
        st.title("Research Assistant")

    init_session_state()
    documentRetrieval = DocumentRetrieval()
    
    mode = st.sidebar.radio("Choose Mode", ["Document Chat", "Web Search"])
    
    with st.sidebar:
        st.title("Setup")
        st.markdown("**1. Choose your data source**")
        
        handle_document_upload(documentRetrieval)
        
        if mode == "Web Search":
            st.markdown("Web search mode selected. Documents will be used as additional context.")
        
        with st.expander("Additional settings", expanded=True):
            st.markdown("**Interaction options**")
            st.markdown("**Note:** Toggle these at any time to change your interaction experience")
            show_sources = st.checkbox("Show sources", value=True, key="show_sources")
            
            st.markdown("**Reset chat**")
            st.markdown("**Note:** Resetting the chat will clear all conversation history")
            if st.button("Reset conversation"):
                st.session_state.conversation = None
                st.session_state.chat_history = []
                st.session_state.sources_history = []
                st.session_state.search_assistant = None
                st.toast("Conversation reset. The next response will clear the history on the screen")
    
    user_question = st.chat_input("Ask a question", disabled=st.session_state.input_disabled)
    handle_user_input(user_question, mode)

if __name__ == "__main__":
    main()

