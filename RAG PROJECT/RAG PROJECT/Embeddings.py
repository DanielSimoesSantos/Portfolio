# Databricks notebook source
!pip install sentence_transformers 
!pip install orjson 
!pip install -q chromadb
!pip install pydantic==1.10.7
!pip install openai
!pip install google-generativeai

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import orjson   
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai


# COMMAND ----------

jsonl_path = "/Workspace/Users/a3702244-msp01@ey.net/RAG PROJECT/Chunks/thesis_chunks_pymudf.jsonl"

# Load texts + metadata
texts, metadatas, ids = [], [], []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        rec = orjson.loads(line)
        texts.append(rec["text"])
        metadatas.append({
            "doc_id": rec.get("doc_id"),
            "page_start": rec.get("page_start"),
            "page_end": rec.get("page_end"),
        })
        ids.append(rec.get("chunk_id"))

# COMMAND ----------

#Instruction‑tuned embedding model (E5 large)
model = SentenceTransformer("intfloat/e5-large-v2")

# COMMAND ----------

# Embed passages with the correct prefix
passage_inputs = [f"passage: {t}" for t in texts]
embeddings = model.encode(
    passage_inputs,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(embeddings.shape)

# COMMAND ----------

client = chromadb.PersistentClient(path="/Workspace/Users/a3702244-msp01@ey.net/RAG PROJECT/Vector DB") 
collection = client.get_or_create_collection(
    name="thesis_chunks",
    metadata={"hnsw:space": "cosine"}    # cosine space for normalized vectors
)

# Add all records
collection.add(
    embeddings=embeddings.astype("float32").tolist(),
    documents=texts,
    metadatas=metadatas,
    ids=ids
)

# COMMAND ----------

def retrieve_chunks(question: str, n_results: int = 5):
    query_embedding = model.encode(
        [f"query: {question}"],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

    retrieved = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
        retrieved.append({
            "chunk_id": chunk_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "similarity": 1 - dist  # cosine similarity if using cosine distance
        })

    return retrieved

# COMMAND ----------

def build_context(retrieved_chunks):
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk["metadata"] or {}
        doc_id = meta.get("doc_id", "unknown_doc")
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")

        block = (
            f"[Source {i}] "
            f"(doc_id={doc_id}, pages={page_start}-{page_end}, "
            f"chunk_id={chunk['chunk_id']}, sim={chunk['similarity']:.3f})\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

# COMMAND ----------

def build_prompt(question: str, context: str):
    return f"""
You are a thesis assistant.

Answer the question using ONLY the provided context.
If the context is not enough, say clearly: "I could not find enough evidence in the retrieved thesis chunks."
Do not invent facts.
When possible, cite the source blocks like [Source 1], [Source 2].

Question:
{question}

Context:
{context}

Answer:
""".strip()

# COMMAND ----------

os.environ["GOOGLE_API_KEY"] = "AIzaSyCbN3MW-2eKrpbUASrOZRjx3HxFX2kFLmQ"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def generate_answer(prompt, model="gemini-2.5-flash"):
    llm = genai.GenerativeModel(model)

    response = llm.generate_content({
        "parts": [
            {"text": "You are a helpful research assistant answering questions about a thesis."},
            {"text": prompt}
        ]
    })

    return response.text

# COMMAND ----------

def ask_rag(question: str, n_results: int = 5):
    retrieved_chunks = retrieve_chunks(question, n_results=n_results)
    context = build_context(retrieved_chunks)
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "prompt": prompt,
        "answer": answer
    }

# COMMAND ----------

question = "What methodology did the thesis use to evaluate the model?"

result = ask_rag(question, n_results=5)

print("QUESTION:")
print(result["question"])
print("\n" + "="*80 + "\n")

print("RETRIEVED CHUNKS:")
for i, chunk in enumerate(result["retrieved_chunks"], start=1):
    meta = chunk["metadata"] or {}
    print(
        f"#{i} | chunk_id={chunk['chunk_id']} | "
        f"pages {meta.get('page_start')}–{meta.get('page_end')} | "
        f"sim={chunk['similarity']:.3f}"
    )
    print(chunk["text"][:300].replace("\n", " "), "...\n")

print("\n" + "="*80 + "\n")
print("FINAL ANSWER:")
print(result["answer"])