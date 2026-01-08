# ===============================
# SIMPLE RAG FROM PDF (PYTORCH ONLY)
# ===============================

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# -------------------------------
# 1. LOAD PDF AND EXTRACT TEXT
# -------------------------------
def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -------------------------------
# 2. SPLIT TEXT INTO CHUNKS
# -------------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():

    # PDF FILE
    pdf_path = "sample.pdf"   # <-- Put your PDF here
    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    # -------------------------------
    # 3. CREATE EMBEDDINGS (PyTorch)
    # -------------------------------
    print("Loading embedding model (MiniLM)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # ~22M parameters, CPU friendly

    print("Creating embeddings for chunks...")
    chunk_embeddings = embed_model.encode(chunks)

    # -------------------------------
    # 4. STORE IN VECTOR DATABASE (FAISS)
    # -------------------------------
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))
    print("Vector store created.")

    # -------------------------------
    # 5. USER QUESTION
    # -------------------------------
    query = input("\nAsk a question: ")
    query_embedding = embed_model.encode([query])

    # -------------------------------
    # 6. SIMILARITY SEARCH (Top-K)
    # -------------------------------
    k = 3  # Top 3 chunks
    distances, indices = index.search(np.array(query_embedding), k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\nChunk {i}:\n{chunk}")

    # -------------------------------
    # 7. LOAD SMALL LLM (PyTorch)
    # -------------------------------
    print("\nLoading small language model (flan-t5-small)...")
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=200
    )

    # -------------------------------
    # 8. GENERATE FINAL ANSWER
    # -------------------------------
    context = "\n".join(retrieved_chunks)
    prompt = f"""
Context:
{context}

Question:
{query}

Answer using only the context.
"""

    result = llm(prompt)
    print("\n================ FINAL ANSWER ================")
    print(result[0]["generated_text"])
    print("============================================")

# -------------------------------
# RUN MAIN
# -------------------------------
if __name__ == "__main__":
    main()
