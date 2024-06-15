import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    os.environ['HUGGINGFACEHUB_API_TOKEN']="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"
    # Instantiate the LLM model
    llm = HuggingFaceHub(repo_id='facebook/bart-large-cnn', model_kwargs={'temperature':1.5})
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='🦜 Text Summarization App')
st.title('🦜 Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    huggingface_api_key = st.text_input('HuggingFace API Key', type = 'password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted and huggingface_api_key.startswith('hf_'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            del huggingface_api_key

if len(result):
    st.info(response)