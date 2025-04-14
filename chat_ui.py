import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# 安全地讀取 OpenAI API 金鑰（由 Streamlit secrets 提供）
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 初始化向量資料庫，使用快取避免重複加載
@st.cache_resource
def setup_vectorstore():
    loader = TextLoader("faq.txt", encoding="utf-8")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    return FAISS.from_documents(split_docs, embedding)

# 初始化 LLM 與 QA Chain
vectorstore = setup_vectorstore()
llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

# Streamlit UI
st.set_page_config(page_title="AI 校務問答機器人 🤖")
st.title("🤖 AI 校務客服 ChatBot")
query = st.text_input("請輸入你的問題")

if query:
    docs = vectorstore.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    st.write("### 回答：")
    st.success(answer)
