# === INSTALL EVERYTHING ===
!pip install -q pdfplumber pytesseract pillow \
langchain langchain-community \
transformers accelerate bitsandbytes \
faiss-cpu sentence-transformers huggingface_hub

# === LOGIN TO HUGGING FACE ===
from huggingface_hub import notebook_login
notebook_login()


import pdfplumber, pytesseract, io
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from google.colab import files


from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import HfFolder
import torch