import os
import shutil
import streamlit as st
import time
from dotenv import load_dotenv
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from models.model_loader import load_model
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from byaldi import RAGMultiModalModel
import json
import uuid
import numpy as np
import base64  # For image encoding

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

UPLOAD_FOLDER = 'uploaded_documents'
INDEX_FOLDER = '.byaldi'
SESSION_FOLDER = 'sessions'
STATIC_FOLDER = 'static/images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize Streamlit
st.set_page_config(page_title="Vision & Text RAG Chatbot", layout="wide")
st.title("🤖 Vision & Text RAG Chatbot")

# Session State Initialization
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
    st.session_state['index_name'] = None
    st.session_state['rag_model'] = None
    st.session_state['conversation_chain'] = None

# Helper Functions
def clear_uploaded_files():
    folders_to_clear = [UPLOAD_FOLDER, INDEX_FOLDER, SESSION_FOLDER]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)

def get_base64_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def vector_embedding():
    if "vectors" not in st.session_state:
        # Display status messages in the sidebar
        st.sidebar.write("Initializing embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        st.sidebar.write("Loading documents from PDF directory...")
        loader = PyPDFDirectoryLoader("./uploaded_documents")
        docs = loader.load()

        st.sidebar.write("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_documents = text_splitter.split_documents(docs)

        st.sidebar.write("Creating vector embeddings...")
        batch_size = 100
        all_embeddings = []
        total_batches = len(final_documents) // batch_size + (1 if len(final_documents) % batch_size != 0 else 0)

        # Initialize progress bar in the sidebar
        progress_bar = st.sidebar.progress(0)

        with st.spinner("Creating vector embeddings..."):
            for i in range(0, len(final_documents), batch_size):
                batch = final_documents[i:i+batch_size]
                texts = [doc.page_content for doc in batch]
                batch_embeddings = embeddings.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)
                progress = (i // batch_size + 1) / total_batches
                progress_bar.progress(progress)
                time.sleep(0.1)  # Simulate time-consuming embedding

        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        texts = [doc.page_content for doc in final_documents]
        metadatas = [doc.metadata for doc in final_documents]

        st.session_state.vectors = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array)),
            embedding=embeddings,
            metadatas=metadatas
        )

        st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectors)
        st.sidebar.success("Vector embeddings created.")

def get_conversation_chain(vectorstore):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama-3.1-8b-instant')
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )
    return conversation_chain

# Custom CSS for Scrollable Containers
st.markdown(
    """
    <style>
    .scrollable-container {
        height: 500px; /* Fixed height to enable scrolling */
        overflow-y: auto; /* Enable vertical scrolling */
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
    }
    /* Optional: Adjust scrollbar appearance */
    .scrollable-container::-webkit-scrollbar {
        width: 8px;
    }
    .scrollable-container::-webkit-scrollbar-thumb {
        background-color: #aaa;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Components
with st.sidebar:
    # Optional: Add a logo if available
    logo_path = os.path.join(STATIC_FOLDER, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, DOCX, Images)", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'png', 'jpg', 'jpeg']
    )
    index_btn = st.button("📤 Index Documents")

    with st.expander("⚙️ Settings"):
        clear_index = st.button("🧹 Clear Indexed Data")
        if clear_index:
            clear_uploaded_files()
            st.success("All uploaded files and indexes have been cleared.")

# Handling Document Indexing
if index_btn:
    if uploaded_files:
        with st.spinner("Uploading and indexing documents..."):
            for file in uploaded_files:
                file_path = os.path.join(UPLOAD_FOLDER, file.name)
                with open(file_path, 'wb') as f:
                    f.write(file.read())
            try:
                index_documents(
                    UPLOAD_FOLDER, 
                    index_name=st.session_state['session_id'], 
                    index_path=INDEX_FOLDER, 
                    indexer_model='vidore/colpali'
                )
                vector_embedding()
                st.sidebar.success("Documents indexed and embedded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error during indexing: {str(e)}")
    else:
        st.sidebar.warning("Please upload at least one file before indexing.")

# Main Content Area
st.header("💬 Query the Chatbot")
query = st.text_area("Enter your query here:", height=100)

if st.button("🔍 Get Response"):
    if st.session_state.get('conversation_chain'):
        if query.strip() == "":
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    # Vision RAG Agent
                    rag_model = RAGMultiModalModel.from_index(st.session_state['session_id'])
                    vision_images = retrieve_documents(
                        rag_model, query, st.session_state['session_id'], k=3
                    )
                    vision_response, vision_used_images = generate_response(
                        vision_images, query, st.session_state['session_id'], 
                        model_choice='groq-llama-vision'
                    )

                    # Normal RAG Agent
                    conversation_chain = st.session_state['conversation_chain']
                    response = conversation_chain.invoke({'question': query, 'chat_history': []})
                    normal_response = response['answer']

                    # Consolidate Responses using ChatGroq
                    llm = ChatGroq(groq_api_key=groq_api_key, model_name='llama-3.1-8b-instant')
                    summary_prompt = f"""
                    You are an intelligent assistant. Here are two responses to the user's query:

                    **Vision RAG Response:**  
                    {vision_response}

                    **Normal RAG Response:**  
                    {normal_response}

                    Please summarize these responses into a coherent and concise answer.
                    """
                    final_summary = llm.invoke(summary_prompt)  # Generate the final summary

                    # Create two equal columns for response and images
                    response_col, images_col = st.columns([1, 1])

                    with response_col:
                        st.subheader("📝 Consolidated Response")
                        st.markdown(f"{final_summary.content}")

                    with images_col:
                        st.subheader("🖼️ Extracted Images")
                        if vision_used_images:
                            # Begin the scrollable container
                            images_html = '<div class="scrollable-container">'
                            for img_path in vision_used_images:
                                img_full_path = os.path.join(STATIC_FOLDER, os.path.relpath(img_path, 'static/images'))
                                if os.path.exists(img_full_path):
                                    # Encode image to base64
                                    img_base64 = get_base64_image(img_full_path)
                                    ext = os.path.splitext(img_full_path)[1].lower()
                                    mime = f"image/{ext[1:]}" if ext in ['.png', '.jpg', '.jpeg', '.gif'] else "image/png"
                                    # Append the image HTML
                                    images_html += f'<img src="data:{mime};base64,{img_base64}" alt="Relevant Image" style="width:100%; margin-bottom:10px;">'
                                else:
                                    images_html += f'<p>Image not found: {img_full_path}</p>'
                            # Close the scrollable container
                            images_html += '</div>'
                            # Render the images within the scrollable container
                            st.markdown(images_html, unsafe_allow_html=True)
                        else:
                            st.info("No relevant images were extracted.")
                except Exception as e:
                    st.error(f"Error processing the query: {str(e)}")
    else:
        st.warning("Please index and embed documents first by uploading and indexing your documents.")

# Footer
st.markdown("---")
st.markdown("© 2024 Vision & Text RAG Chatbot. All rights reserved.")
