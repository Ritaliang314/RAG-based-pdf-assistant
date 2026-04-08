import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

def extract_text_from_pdf(pdf_file) -> str:
    """從 PDF 擷取文字"""
    pdf_reader = PdfReader(pdf_file)
    text = []

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text.append(content)

    return "\n".join(text)


def build_vectorstore(text: str, api_key: str):
    """將文字切塊後建立向量資料庫"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    # 去掉空白 chunk
    chunks = [c.strip() for c in chunks if c.strip()]

    if not chunks:
        return None

    # 可先限制前幾百段，避免超大 PDF 一次打爆
    chunks = chunks[:200]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    try:
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"建立向量庫失敗：{type(e).__name__}: {e}")
        st.stop()


def build_rag_chain(vectorstore, api_key: str):
    """建立 Retrieval QA Chain"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """你是一個 PDF 文件問答助手。
請只能根據提供的文件內容回答問題。
如果文件內容中找不到答案，請明確回答：
「我無法從提供的 PDF 內容中找到答案。」

文件內容：
<context>
{context}
</context>

問題：
{input}

請使用繁體中文回答，並盡量簡潔清楚。"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def reset_state_if_new_file(uploaded_file):
    """若上傳新檔案，重置 session state"""
    current_file_name = uploaded_file.name if uploaded_file else None

    if st.session_state.get("current_file_name") != current_file_name:
        st.session_state.current_file_name = current_file_name
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.pdf_text = None


def main():
    load_dotenv(".env")

    st.set_page_config(page_title="Ask your PDF", page_icon="💬")
    st.header("Ask your PDF 💬 (Powered by Gemini)")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.info("請確認有給予 GOOGLE_API_KEY")
        #api_key = st.text_input("Google API Key", type="password")
        if not api_key:
            st.stop()

    uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_pdf is None:
        st.caption("請先上傳一個 PDF 檔案。")
        st.stop()

    reset_state_if_new_file(uploaded_pdf)

    if st.session_state.vectorstore is None:
        with st.spinner("正在讀取 PDF 並建立知識庫..."):
            text = extract_text_from_pdf(uploaded_pdf)

            if not text.strip():
                st.error("無法從 PDF 擷取文字。這份 PDF 可能是掃描檔、圖片型 PDF，或內容不可抽取。")
                st.stop()

            vectorstore = build_vectorstore(text, api_key)
            if vectorstore is None:
                st.error("PDF 內容切分失敗，無法建立知識庫。")
                st.stop()

            rag_chain = build_rag_chain(vectorstore, api_key)

            st.session_state.pdf_text = text
            st.session_state.vectorstore = vectorstore
            st.session_state.rag_chain = rag_chain

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        with st.spinner("Gemini 思考中..."):
            result = st.session_state.rag_chain.invoke({"input": user_question})

        st.subheader("回答：")
        st.write(result["answer"])


if __name__ == "__main__":
    main()