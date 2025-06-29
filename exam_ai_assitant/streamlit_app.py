!pip install -q streamlit
%%writefile app.py
import streamlit as st

st.title("Document RAG Application")
import os
import io
from embedder import extract_text, chunk_text, embed_chunks
from rag_chain import build_rag_chain
from langchain_community.embeddings import SentenceTransformerEmbeddings

st.set_page_config(page_title="📚 AI Exam Assistant", layout="centered")

def main():
    st.title("📚 AI Exam Assistant")
    st.markdown("Upload a textbook image or PDF and get exam-style questions generated using RAG + LLM.")

    # Initialize embedding model
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
    num_questions = st.slider("Number of questions to generate", 3, 10, 5)

    if uploaded_file:
        try:
            # Read the file into memory
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name

            with st.spinner("🔍 Extracting text and building vector index..."):
                # Extract text, chunk, and embed
                extracted_text = extract_text(image_bytes=file_bytes, file_name=file_name)
                chunks = chunk_text(extracted_text)
                vectorstore = embed_chunks(chunks, embedding_model, index_path="faiss_index") # Pass embedding_model

            if not vectorstore:
                st.error("❌ Failed to create vector index.")
            else:
                with st.spinner("🧠 Generating questions using the LLM..."):
                    rag_chain = build_rag_chain(embedding_model, num_questions=num_questions) # Pass embedding_model
                    if rag_chain:
                        response = rag_chain.invoke({"query": f"Generate {num_questions} questions based on this document."})
                        result = response.get("result", "❌ No questions generated.")
                        st.success("✅ Questions generated!")
                        st.markdown("### 🧠 Exam-Style Questions:")
                        st.markdown(result)
                    else:
                        st.error("❌ Failed to build the RAG chain.")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()