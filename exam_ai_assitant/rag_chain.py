embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vectorstore(path="faiss_index"):
    try:
        return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"⚠️ Vectorstore load failed: {e}")
        return None

def build_rag_chain(num_questions=5):
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token = HfFolder.get_token()

    try:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, local_files_only=True)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            device=device,
            pad_token_id=tokenizer.eos_token_id
        )

        llm = HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        print(f"❌ Model load error: {e}")
        print("Check token access, model name, and RAM availability.")
        return None

    # Prompt with num_questions baked in
    prompt = PromptTemplate(
        input_variables=["context"],
        template=f"""
You are a smart study assistant. A student uploaded this text:

---
{{context}}
---

Now do the following:

1. Generate exactly **{num_questions}** diverse exam-style questions that:
   - Cover different concepts
   - Mix factual, conceptual, and tricky types
   - At least 2 require external knowledge

2. Provide answers and difficulty tags like:
   - (💡 Easy / From Text)
   - (🧠 Requires External Knowledge)
   - (🔥 Likely Exam Question)
"""
    )

    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            document_variable_name="context"
        ),
        return_source_documents=False
    )

# === Run RAG Chain ===
question_count = 6
chain = build_rag_chain(num_questions=question_count)

if chain:
    print("✅ Generating questions...\n")
    response = chain.invoke({"query": f"Generate {question_count} questions from the document"})
    print("📘 Generated:\n\n", response.get("result", "❌ No response generated."))
else:
    print("❌ Failed to build RAG chain.")