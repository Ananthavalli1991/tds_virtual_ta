from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import base64
import io
import numpy as np
import json
import os
from openai import OpenAI
import uvicorn

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust if needed

# Load OpenAI client
client = OpenAI(
    api_key=os.getenv("AIPROXY_TOKEN"),
    base_url=os.getenv("OPENAI_BASE_URL")  # e.g. https://aiproxy.sanand.workers.dev/openai/v1
)

app = FastAPI()

# Load content from both sources
with open(r"C:\Users\DELLdiscourse_pages.json", "r", encoding="utf-8") as f:
    course_docs = json.load(f)
with open(r"C:\Users\DELLcourse_pages.json", "r", encoding="utf-8") as f:
    discourse_docs = json.load(f)

all_docs = course_docs + discourse_docs
all_texts = [doc["content"] for doc in all_docs]

# Request schema
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 string
@app.get("/")
def root():
    return {"message": "TDS Virtual TA is up and running"}
@app.post("/api/")
async def rag_api(query: QueryRequest):
    question = query.question.strip()

    # Optional: OCR from base64 image
    if query.image:
        try:
            image_bytes = base64.b64decode(query.image + "===")
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image)
            question += " " + ocr_text.strip()
        except Exception as e:
            print("❌ OCR failed:", e)

    # Embed question + documents
    embed_model = "text-embedding-3-small"
    try:
        embed_inputs = [question] + all_texts
        embeddings = client.embeddings.create(
            input=embed_inputs,
            model=embed_model
        ).data
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}

    q_vec = np.array(embeddings[0].embedding, dtype="float32")
    doc_vecs = [np.array(e.embedding, dtype="float32") for e in embeddings[1:]]

    # Compute cosine similarity
    sims = [np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec)) for vec in doc_vecs]
    top_k = np.argsort(sims)[-5:][::-1]
    top_docs = [all_docs[i] for i in top_k]

    # Construct context
    context = "\n\n".join(doc["content"] for doc in top_docs)

    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful TDS teaching assistant. Use only the context below to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return {"error": f"Chat failed: {str(e)}"}

    # Prepare links
    links = [{"url": doc.get("url"), "text": doc.get("title")} for doc in top_docs if doc.get("url")]

    return {
        "answer": answer,
        "links": links
    }

# Run locally
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
