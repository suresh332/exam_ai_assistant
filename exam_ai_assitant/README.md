# 📚 AI Exam Assistant with RAG + OCR + LangChain + Mistral

This project is a **Retrieval-Augmented Generation (RAG)** based AI assistant that takes a **text input or image of a textbook/study material**, extracts the text using OCR (if needed), chunks and embeds it using FAISS, and then uses a **Large Language Model (LLM)** like **Mistral-7B** to generate **exam-style questions**.

> 🔍 Think of it as a smart tool that reads your notes and prepares a mini test — instantly!

---

## 📸 Demo Preview

> ![Demo Screenshot](https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/images/rag_example.png)  
> _Input: Textbook image or paragraph → Output: Conceptual, Factual, Tricky Questions_

---

## 💡 Key Features

✅ Upload text, scanned images, or textbook pages  
✅ OCR to extract clean text from images (Tesseract)  
✅ Automatic text chunking with overlap  
✅ Embeds using `MiniLM` + FAISS for retrieval  
✅ Uses **Mistral-7B Instruct** for generating:
- 🔹 Conceptual questions  
- 🔹 Applied reasoning  
- 🔹 External knowledge-based questions  
✅ Tags each question with difficulty: 💡🧠🔥  
✅ Built with **Streamlit** for interactive local app

---

## 🧰 Tech Stack

| Component         | Tool/Library                            |
|------------------|-----------------------------------------|
| Embedding Model   | `all-MiniLM-L6-v2` via `sentence-transformers` |
| OCR Engine        | `Tesseract OCR` via `pytesseract`      |
| RAG Framework     | `LangChain`                            |
| Vector DB         | `FAISS`                                |
| LLM               | `mistralai/Mistral-7B-Instruct-v0.1`   |
| Interface         | `Streamlit`                            |

---

## 📂 Project Structure

ai-exam-assistant/
│
├── app/
│ ├── embedder.py # OCR, chunking, and FAISS indexing
│ ├── rag_chain.py # Loads LLM and builds LangChain RAG chain
│ ├── streamlit_app.py # Streamlit UI for end-to-end pipeline
│
├── sample_input/
│ └── sample_text.txt # Sample text-based input
│
├── requirements.txt # All dependencies
├── README.md # Project documentation


📌 Why This Project?
This project demonstrates:

Real-world RAG agent design

Practical use of OCR + NLP + LLM

Hands-on knowledge of LangChain, Hugging Face, and Streamlit

Ability to integrate multiple AI components into a working product

👨‍💻 Author
Suresh Purbia
AI/ML Enthusiast | Deep Learning | LangChain & Transformers
📫 LinkedIn • GitHub
🏁 Future Add-ons
 Streamlit Cloud or Hugging Face deployment

 Multi-language OCR support

 Export Q&A to PDF

 Voice input mode for accessibility

⭐ If you found this useful
Leave a ⭐ on the repo or connect with me on LinkedIn to discuss collaborations and ideas!
