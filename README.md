### AI-Powered News Analysis Engine

This tool is designed to unlock AI-powered insights from your news articles in just seconds. Simply input article URLs, ask questions, and receive actionable information from the stock market and financial domain.
![image](https://github.com/user-attachments/assets/e9c6a2fb-2654-4448-bb8f-d8828cc44769)

### Features
• Simple URL Input: Enter up to three news article URLs for instant processing.

• AI-Powered Insights: Ask specific questions related to the articles and get relevant responses.

• Data Processing: Leverage LangChain's tools and OpenAI embeddings for article content analysis.

• FAISS for Speed: The FAISS library ensures efficient and swift retrieval of indexed information.

• User-Friendly Interface: With a straightforward design, interact directly with the tool and receive actionable insights.

### Usage/Examples
1.Run the Streamlit app by executing:

streamlit run main.py

2.The web app will open in your browser.

•On the sidebar, you can input up to three news article URLs related to the stock market, financial domain, or any relevant topics.

•Click "Process URLs" to start analyzing the content from the provided articles.
The system will:

•Perform text processing and splitting for each article.

•Generate embeddings using an AI model and efficiently index them using FAISS.

•The FAISS index will be stored locally in a pickle file for faster future retrieval.

•You can now ask questions about the content of the articles directly and receive AI-generated insights in seconds.

•Example URLs used during testing:

MoneyControl: Tata Motors & Mahindra certificates

Renish Ladhani: Dolomites Budget Guide

Nature: Research on AI in finance

### Project Structure
•main.py: The main Streamlit application script for running the app.

•requirements.txt: A list of required Python packages, including libraries for text processing and FAISS.

•faiss_store.pkl: A pickle file that stores the FAISS index for embedding vectors.

•.env: A configuration file for securely storing your OpenAI API key.
