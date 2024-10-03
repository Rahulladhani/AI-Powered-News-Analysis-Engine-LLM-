import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (especially OpenAI API key)
load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Get URLs from the sidebar and clean up quotes
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    # Clean up the URL by stripping any extra quotes
    if url:
        urls.append(url.strip('"'))

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
index_directory = "vector_index_data"  # Directory to save FAISS index

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
    except Exception as e:
        main_placeholder.text(f"Error loading URLs: {str(e)}")
        st.stop()  # Stop execution if there is an error

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and build FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index locally (no pickling required)
    vectorstore_openai.save_local(index_directory)
    main_placeholder.text("FAISS index saved locally...✅✅✅")

# Query input
query = main_placeholder.text_input("Question: ")

if query:
    # Load the FAISS index from disk
    if os.path.exists(index_directory):
        embeddings = OpenAIEmbeddings()  # Ensure embeddings are initialized
        try:
            # Load FAISS index with deserialization enabled (use this only if the file is trusted)
            vectorstore = FAISS.load_local(index_directory, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            main_placeholder.text(f"Error loading FAISS index: {str(e)}")
            st.stop()  # Stop execution if there is an error

        # Set up the Retrieval QA chain with the loaded vectorstore
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display the sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split sources by newline
            for source in sources_list:
                st.write(source)
