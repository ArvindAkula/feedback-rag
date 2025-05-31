import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Call set_page_config as the very first Streamlit command
st.set_page_config(page_title="Review Insights Q&A", layout="wide")

# --- Configuration ---
MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model
CSV_FILE_PATH = "data/reviews.csv"       # Path to your reviews CSV
OPENAI_MODEL = "gpt-4" # OpenAI model for answer generation (e.g., "gpt-4", "gpt-3.5-turbo")
# Ensure you have a .env file in the same directory as this script, or set environment variables
# The .env file should contain: OPENAI_API_KEY=your_openai_api_key

# --- Caching Functions ---

@st.cache_resource
def get_openai_client(api_key_to_use: str):
    """Loads the OpenAI API key and returns an OpenAI client instance."""
    if not api_key_to_use:
        st.error("API key not provided to OpenAI client initializer.")
        return None
    # load_dotenv() # Still useful if other parts of app might use .env vars independently
    # openai_key = os.getenv("OPENAI_API_KEY") # We are now passing the key directly
    # if not openai_key:
    #     st.error("OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable.")
    #     return None
    return OpenAI(api_key=api_key_to_use)

@st.cache_resource
def load_sentence_transformer_model(model_name: str):
    """Loads and caches the SentenceTransformer model."""
    with st.spinner(f"Loading sentence embedding model ({model_name})... This may take a moment on the first run."):
        model = SentenceTransformer(model_name)
    return model

@st.cache_data
def load_and_preprocess_chunks(file_path: str) -> list:
    """Loads reviews from CSV, preprocesses, and splits them into chunks. Caches the chunks."""
    st.info(f"Attempting to load reviews from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The review file '{file_path}' was not found. Please ensure it's in the correct location.")
        return []
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return []

    if "review_text" not in df.columns:
        st.error("Error: 'review_text' column not found in the CSV file.")
        return []

    df = df.dropna(subset=["review_text"])
    if df.empty:
        st.warning("No data found in 'review_text' column after dropping empty rows, or the file is empty.")
        return []

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=50   # Overlap between chunks to maintain context
    )

    all_text_chunks = []
    for text in df["review_text"]:
        if pd.notna(text) and isinstance(text, str) and text.strip():
            chunks_list = character_splitter.split_text(text)
            all_text_chunks.extend(chunks_list)

    if not all_text_chunks:
        st.warning("No text chunks were generated from the review data. The source file might be empty or contain no processable text.")
    else:
        st.info(f"Successfully processed {len(df)} reviews into {len(all_text_chunks)} text chunks.")
    return all_text_chunks

@st.cache_data
def get_embeddings(_chunks: list, _model_name_for_cache_key: str) -> np.ndarray:
    """Generates and caches embeddings for the given text chunks using the specified model."""
    if not _chunks:
        return np.array([])

    s_model = load_sentence_transformer_model(_model_name_for_cache_key) # Fetches cached SentenceTransformer model

    with st.spinner(f"Generating embeddings for {len(_chunks)} chunks... This can take some time on the first run."):
        try:
            embeddings_array = s_model.encode(_chunks, show_progress_bar=False) # Set to True for console progress if needed
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return np.array([])
    return embeddings_array

# --- Core Logic Functions (adapted for Streamlit) ---

def retrieve_relevant_chunks_streamlit(query: str, all_doc_chunks: list, doc_embeddings: np.ndarray, s_model: SentenceTransformer, k=5) -> list:
    """Retrieves the top k most relevant text chunks for a given query."""
    if not all_doc_chunks or doc_embeddings.size == 0:
        st.warning("No review data or embeddings available for search.")
        return []
    try:
        query_embedding = s_model.encode([query])
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_k_actual = min(k, len(all_doc_chunks))
        top_k_idx = np.argsort(scores)[-top_k_actual:][::-1]
        return [all_doc_chunks[i] for i in top_k_idx]
    except Exception as e:
        st.error(f"Error during relevance search: {e}")
        return []


def generate_answer_from_chunks_streamlit(query: str, relevant_doc_chunks: list, openai_client_instance: OpenAI) -> str:
    """Generates an answer to the query based on provided chunks using OpenAI."""
    if not relevant_doc_chunks:
        return "No relevant information was found in the reviews to answer this query."

    context = "\n\n---\n\n".join(relevant_doc_chunks)

    prompt_message = f"""Please answer the following question based *only* on the provided text excerpts from customer reviews.
If the answer cannot be found in the excerpts, please state that. Be concise and direct.

Question: "{query}"

Excerpts:
{context}

Answer:"""

    try:
        response = openai_client_instance.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided text excerpts from customer reviews."},
                {"role": "user", "content": prompt_message}
            ],
            temperature=0.0 # For factual answers
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while communicating with OpenAI: {e}")
        return "Sorry, I encountered an error while trying to generate an answer from OpenAI."

# --- API Key Handling & OpenAI Client Initialization ---
# Attempt to get API key from environment variables first
openai_api_key_global = os.getenv("OPENAI_API_KEY")

# If not found in environment, prompt user in sidebar
if not openai_api_key_global:
    st.sidebar.markdown("### 🔑 Paste your OpenAI API Key")
    openai_api_key_global = st.sidebar.text_input(
        label="OpenAI API Key (sk-...)", # Clearer label
        type="password",
        placeholder="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        help="You can find your API key on the OpenAI dashboard."
    )

# If no API key is provided after prompting, stop the app
if not openai_api_key_global:
    st.error("⚠️ Please provide an OpenAI API key (in the sidebar) to continue.")
    st.info("You can obtain an API key from the OpenAI website.")
    st.stop()

# Initialize OpenAI client globally using the obtained API key
openai_client_global = get_openai_client(openai_api_key_global)

# If client initialization fails (e.g., invalid key format, though OpenAI client might not check this on init), stop.
if not openai_client_global:
    st.error("OpenAI client could not be initialized. Please check your API key and ensure it is correct.")
    st.stop()

# --- Streamlit App UI ---
st.title("📄 Review Insights Q&A") # Using the title with emoji as it was later in the script
st.markdown("Ask questions about customer reviews and get answers powered by AI.")


# --- Initialization of data and models ---
s_model_global = load_sentence_transformer_model(MODEL_NAME)
all_chunks_global = load_and_preprocess_chunks(CSV_FILE_PATH)

if not all_chunks_global:
    st.warning("No review chunks available to process. Please check the CSV file and its content at '{CSV_FILE_PATH}'.")
    st.stop()

embeddings_global = get_embeddings(all_chunks_global, MODEL_NAME)

if embeddings_global.size == 0 and all_chunks_global:
    st.error("Failed to generate embeddings for the review data. The app cannot proceed.")
    st.stop()
elif embeddings_global.size > 0:
    st.success(f"Successfully loaded and processed {len(all_chunks_global)} review chunks. Ready for your questions!")
else:
    st.info("Waiting for review data to be processed or review data is empty.")
    st.stop()


# --- Main Interaction Area ---
st.markdown("---")
user_query = st.text_input("Enter your question about the reviews:", placeholder="e.g., What are common complaints about app performance?")

if st.button("Get Answer", type="primary", use_container_width=True):
    if user_query:
        with st.spinner("Searching reviews and crafting your answer..."):
            retrieved_chunks = retrieve_relevant_chunks_streamlit(
                user_query,
                all_chunks_global,
                embeddings_global,
                s_model_global,
                k=10 # Number of chunks to retrieve for context
            )

            if retrieved_chunks:
                st.subheader("💡 AI-Generated Answer:")
                answer = generate_answer_from_chunks_streamlit(user_query, retrieved_chunks, openai_client_global)
                st.markdown(answer)

                with st.expander("View relevant review excerpts used for this answer"):
                    for i, chunk_text in enumerate(retrieved_chunks):
                        st.markdown(f"**Excerpt {i+1}:**")
                        st.markdown(f"> _{chunk_text}_")
            else:
                st.info("Could not find any relevant review excerpts for your query in the provided data.")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.caption("Powered by SentenceTransformers, OpenAI, and Streamlit.")