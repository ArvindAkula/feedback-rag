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
# CSV_FILE_PATH = "data/reviews.csv" # No longer needed, will be uploaded
OPENAI_MODEL = "gpt-4" # OpenAI model for answer generation

# --- Caching Functions ---

@st.cache_resource
def get_openai_client(api_key_to_use: str):
    """Returns an OpenAI client instance using the provided API key."""
    if not api_key_to_use:
        st.error("API key not provided to OpenAI client initializer.")
        return None
    return OpenAI(api_key=api_key_to_use)

@st.cache_resource
def load_sentence_transformer_model(model_name: str):
    """Loads and caches the SentenceTransformer model."""
    with st.spinner(f"Loading sentence embedding model ({model_name})... This may take a moment."):
        try:
            model = SentenceTransformer(model_name)
            return model
        except Exception as e:
            st.error(f"Error loading sentence transformer model '{model_name}': {e}")
            return None


@st.cache_data # Cache based on the content of the uploaded file
def load_and_preprocess_chunks(uploaded_file_object) -> list:
    """Loads reviews from an uploaded CSV file, preprocesses, and splits them into chunks."""
    if uploaded_file_object is None:
        return []

    st.info(f"Attempting to load reviews from uploaded file: {uploaded_file_object.name}")
    try:
        # Read the CSV directly from the uploaded file object
        df = pd.read_csv(uploaded_file_object)
    except Exception as e:
        st.error(f"Error loading or parsing CSV file: {e}")
        st.warning("Please ensure the uploaded file is a valid CSV.")
        return []

    if "review_text" not in df.columns:
        st.error("Error: 'review_text' column not found in the uploaded CSV file.")
        st.warning("Please ensure your CSV has a column named 'review_text' containing the review texts.")
        return []

    df = df.dropna(subset=["review_text"])
    if df.empty:
        st.warning("No data found in 'review_text' column after dropping empty rows, or the 'review_text' column is empty.")
        return []

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=50
    )

    all_text_chunks = []
    for text in df["review_text"]:
        if pd.notna(text) and isinstance(text, str) and text.strip():
            chunks_list = character_splitter.split_text(text)
            all_text_chunks.extend(chunks_list)

    if not all_text_chunks:
        st.warning("No text chunks were generated from the review data. The 'review_text' column might contain no processable text.")
    else:
        st.info(f"Successfully processed {len(df)} reviews into {len(all_text_chunks)} text chunks from '{uploaded_file_object.name}'.")
    return all_text_chunks

@st.cache_data # Cache based on the chunks and model name
def get_embeddings(_chunks: list, _model_name_for_cache_key: str) -> np.ndarray:
    """Generates and caches embeddings for the given text chunks."""
    if not _chunks:
        return np.array([])

    # This will use the cached model if called with the same model_name
    s_model = load_sentence_transformer_model(_model_name_for_cache_key)
    if not s_model:
        st.error("Sentence embedding model not available for generating embeddings.")
        return np.array([])

    with st.spinner(f"Generating embeddings for {len(_chunks)} chunks... This can take some time."):
        try:
            embeddings_array = s_model.encode(_chunks, show_progress_bar=False)
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return np.array([])
    return embeddings_array

# --- Core Logic Functions ---

def retrieve_relevant_chunks_streamlit(query: str, all_doc_chunks: list, doc_embeddings: np.ndarray, s_model: SentenceTransformer, k=5) -> list:
    """Retrieves the top k most relevant text chunks for a given query."""
    if not all_doc_chunks or doc_embeddings.size == 0:
        # This state should ideally be handled before calling this function
        st.warning("No review data or embeddings available for search.")
        return []
    if not s_model:
        st.error("Sentence model not available for relevance search.")
        return []
    try:
        query_embedding = s_model.encode([query])
        scores = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_k_actual = min(k, len(all_doc_chunks))
        # Ensure no negative indexing if top_k_actual is 0
        if top_k_actual == 0:
            return []
        top_k_idx = np.argsort(scores)[-top_k_actual:][::-1]
        return [all_doc_chunks[i] for i in top_k_idx]
    except Exception as e:
        st.error(f"Error during relevance search: {e}")
        return []


def generate_answer_from_chunks_streamlit(query: str, relevant_doc_chunks: list, openai_client_instance: OpenAI) -> str:
    """Generates an answer to the query based on provided chunks using OpenAI."""
    if not relevant_doc_chunks:
        return "No relevant information was found in the reviews to answer this query."
    if not openai_client_instance:
        return "OpenAI client is not available to generate an answer."

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
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while communicating with OpenAI: {e}")
        return "Sorry, I encountered an error while trying to generate an answer from OpenAI."

# --- API Key Handling & OpenAI Client Initialization ---
openai_api_key_global = os.getenv("OPENAI_API_KEY")
if not openai_api_key_global:
    st.sidebar.markdown("### 🔑 Paste your OpenAI API Key")
    openai_api_key_global = st.sidebar.text_input(
        label="OpenAI API Key (sk-...)",
        type="password",
        placeholder="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        help="You can find your API key on the OpenAI dashboard."
    )

if not openai_api_key_global:
    st.error("⚠️ Please provide an OpenAI API key (in the sidebar) to continue.")
    st.info("You can obtain an API key from the OpenAI website.")
    st.stop()

openai_client_global = get_openai_client(openai_api_key_global)
if not openai_client_global:
    st.error("OpenAI client could not be initialized. Please check your API key and ensure it is correct.")
    st.stop()

# Load Sentence Transformer model (can be done once, early)
s_model_global = load_sentence_transformer_model(MODEL_NAME)
if not s_model_global:
    st.error("Sentence embedding model could not be loaded. The application cannot proceed.")
    st.stop()

st.title("📄 Review Insights Q&A")
st.markdown("Upload your customer review CSV file and ask questions to get AI-powered insights.")

# File uploader in the sidebar
st.sidebar.markdown("#### Upload Reviews Data") # Added a subheader for clarity
uploaded_file = st.sidebar.file_uploader(
    "Select your CSV file:", # Slightly more instructive label
    type=["csv"],
    help="The CSV file needs a header row. Crucially, one column must be named 'review_text' and contain the customer review texts. Other columns are allowed but will be ignored for the Q&A."
)
st.sidebar.caption("Tip: Ensure your CSV has a column named `review_text` for the Q&A to work correctly.")


all_chunks_global = []
embeddings_global = np.array([])
data_ready = False

if uploaded_file is not None:
    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")
    # Process the uploaded file
    all_chunks_global = load_and_preprocess_chunks(uploaded_file)

    if all_chunks_global: # If chunks were successfully created
        embeddings_global = get_embeddings(all_chunks_global, MODEL_NAME)
        if embeddings_global.size > 0:
            st.success(f"Successfully processed {len(all_chunks_global)} review chunks from '{uploaded_file.name}'. Ready for your questions!")
            data_ready = True
        else:
            st.error("Failed to generate embeddings for the review data from the uploaded file. Please check the file content.")
    else:
        st.warning("No processable review chunks found in the uploaded file. Please ensure it contains valid data in the 'review_text' column.")
else:
    st.info("👈 Please upload your reviews CSV file using the sidebar to get started.")


# --- Main Interaction Area (only if data is ready) ---
if data_ready:
    st.markdown("---")
    user_query = st.text_input("Enter your question about the reviews:", placeholder="e.g., What are common complaints about app performance?")

    if st.button("Get Answer", type="primary", use_container_width=True):
        if user_query:
            if not all_chunks_global or embeddings_global.size == 0 or not s_model_global or not openai_client_global:
                st.error("Required data or models are not loaded. Please ensure file is uploaded and API key is set.")
            else:
                with st.spinner("Searching reviews and crafting your answer..."):
                    retrieved_chunks = retrieve_relevant_chunks_streamlit(
                        user_query,
                        all_chunks_global,
                        embeddings_global,
                        s_model_global,
                        k=10
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
else:
    if uploaded_file is not None: # If a file was uploaded but processing failed
        st.warning("Data from the uploaded file could not be prepared for Q&A. Please check any error messages above.")

st.markdown("---")
st.caption("Powered by SentenceTransformers, OpenAI, and Streamlit.")