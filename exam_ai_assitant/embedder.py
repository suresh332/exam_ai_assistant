embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()

def extract_text(file_path=None, image_bytes=None, file_name=None):
    if file_path:
        if file_path.lower().endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(file_path)
            with io.BytesIO() as output:
                image.save(output, format=image.format)
                image_bytes = output.getvalue()
            return extract_text_from_image(image_bytes)
        else:
            raise ValueError("Unsupported file format")
    elif image_bytes and file_name:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            return extract_text_from_image(image_bytes)
        else:
            raise ValueError("Unsupported file format for uploaded image")
    else:
        raise ValueError("Either file_path or image_bytes and file_name must be provided")

def chunk_text(text, size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)

def embed_chunks(chunks, index_path="faiss_index"):
    try:
        vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
        vectorstore.save_local(index_path)
        return vectorstore
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

# === UPLOAD FILE ===
uploaded = files.upload()
for file_name, file_contents in uploaded.items():
    try:
        text = extract_text(image_bytes=file_contents, file_name=file_name)
        chunks = chunk_text(text)
        store = embed_chunks(chunks)
        if store:
            print(f"✅ Successfully indexed: {file_name}")
        else:
            print("❌ Failed to create vector index.")
    except Exception as e:
        print(f"❌ Error processing file {file_name}: {e}")
