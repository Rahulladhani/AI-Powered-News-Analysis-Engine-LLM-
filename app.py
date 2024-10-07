import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from streamlit import header

#from dotenv import load_dotenv
# Load environment variables (especially OpenAI API key)
#load_dotenv()
headers = {
    "authorization": st.secrets["OPENAI_API_KEY"]
}

st.set_page_config(page_title="AI-Powered News Analysis Engine", page_icon=":newspaper:")

# Page title and tagline
st.title("AI-Powered News Analysis Engine 📰")
st.markdown("### Unlock AI-powered insights from your news articles in seconds.")
st.write("Enter up to three news article URLs, and ask any question related to their content.")

# Sidebar layout for URL input with improved design
st.sidebar.title("Enter News Article URLs Below")

# Add custom CSS for background color and button styling
st.sidebar.markdown(
    """
    <style>
    .sidebar-section {
        background-color: #e1e9f2;
        padding: 10px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #5cb85c;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 14px;
    }
    .stButton>button:hover {
        background-color: #4cae4c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Group the input fields inside a styled box
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("#### Enter URLs to start processing:")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"🔗 URL {i + 1}", placeholder="https://example.com/article")
    if url:
        urls.append(url.strip('"'))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Button to process URLs
process_url_clicked = st.sidebar.button("🔍 **Process URLs**")

# Placeholder for the main content
main_placeholder = st.empty()

# Directory to save FAISS index
index_directory = "vector_index_data"

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

# Progress bar to show during URL processing
progress_bar = st.sidebar.progress(0)

# Process URLs when the button is clicked
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
    except Exception as e:
        main_placeholder.text(f"Error loading URLs: {str(e)}")
        st.stop()

    # Update progress bar
    progress_bar.progress(33)

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting... Started... ✅✅✅")
    docs = text_splitter.split_documents(data)

    # Update progress bar
    progress_bar.progress(66)

    # Create embeddings and build FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vectors... ✅✅✅")
    time.sleep(2)

    # Save the FAISS index locally
    vectorstore_openai.save_local(index_directory)
    main_placeholder.text("FAISS index saved locally... ✅✅✅")

    # Update progress bar to complete
    progress_bar.progress(100)

# Input field for query
query = main_placeholder.text_input("Ask a question about the articles:")

# If the user enters a query, retrieve the answer using the FAISS index
if query:
    if os.path.exists(index_directory):
        embeddings = OpenAIEmbeddings()
        try:
            # Load the FAISS index
            vectorstore = FAISS.load_local(index_directory, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            main_placeholder.text(f"Error loading FAISS index: {str(e)}")
            st.stop()

        # Set up the Retrieval QA chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display the sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

# Footer section (optional)
st.markdown("---")
st.markdown("##### About This Tool")
st.markdown(
    "This tool uses advanced AI models to analyze news articles and extract actionable insights, helping you make informed decisions.")
