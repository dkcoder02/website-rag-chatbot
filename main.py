import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
import os

load_dotenv()

client = OpenAI()

def get_all_links(url):
    """Get all links from the website that belong to the same domain."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        base_domain = urlparse(url).netloc
        
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(url, href)
            if urlparse(full_url).netloc == base_domain:
                links.add(full_url)
        return list(links)
    except Exception as e:
        st.error(f"Error fetching links: {e}")
        return [url]

def process_urls(urls: list[str]):
    """Process multiple URLs and return combined documents."""
    all_docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url])
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_url'] = url
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Error processing {url}: {e}")
    return all_docs

st.title("Chat with Website Content")
st.write("Enter a website URL and ask questions about its content.")

# Replace file uploader with URL input
website_url = st.text_input("Enter website URL", placeholder="https://example.com")

# Process the website URL
if website_url:
    if 'vector_store' not in st.session_state or st.session_state.website_url != website_url:
        st.session_state.website_url = website_url
        
        with st.spinner("Fetching website content..."):
            all_links = get_all_links(website_url)
            st.write(f"Found {len(all_links)} pages to process")

            # Process all URLs
            docs = process_urls(all_links)

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            split_docs = text_splitter.split_documents(documents=docs)

            # Vector Embeddings
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )

            # Create or load vector store
            vector_store = QdrantVectorStore.from_documents(
                url=os.getenv("QDRANT_URL"),
                documents=split_docs,
                prefer_grpc=True,
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="website_vectors",
                embedding=embedding_model
            )

            st.session_state.vector_store = vector_store
            st.success("Website content processed and ready to chat!")

            st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the website content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get vector store from session state
        vector_store = st.session_state.vector_store

        # Vector Similarity Search
        search_results = vector_store.similarity_search(query=prompt)

        # Extract unique source URLs
        source_urls = set()
        for result in search_results:
            if 'source_url' in result.metadata:
                source_urls.add(result.metadata['source_url'])

        context = "\n\n\n".join([f"Content: {result.page_content}\nSource: {result.metadata.get('source_url', 'N/A')}" for result in search_results])

        SYSTEM_PROMPT = f"""
            You are a helpful AI Assistant who answers user queries based on the available context
            retrieved from a website.

            You should only answer the user based on the following context. If the information
            is not available in the context, please say so.

            After your answer, you must include a section titled "Sources:" that lists all the
            URLs where this information was found. Format the URLs as markdown links.
            
            Context: {context}
        """

        # Call OpenAI LLM
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    { "role": "system", "content": SYSTEM_PROMPT },
                    { "role": "user", "content": prompt },
                ]
            )
            response = chat_completion.choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")

else:
    if 'vector_store' in st.session_state:
        del st.session_state.vector_store
    if 'messages' in st.session_state:
        del st.session_state.messages
    if 'website_url' in st.session_state:
        del st.session_state.website_url