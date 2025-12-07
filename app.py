from flask import Flask, request, render_template, jsonify
from pydantic import BaseModel, ValidationError
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# ----------------------
# CONFIGURATION
# ----------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

open_ai_client = OpenAI(api_key=OPENAI_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone_client.Index(INDEX_NAME)

# ----------------------
# FLASK APP
# ----------------------
app = Flask(__name__, static_folder="static", template_folder="static")

# ----------------------
# DATA MODELS
# ----------------------
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

# ----------------------
# EMBEDDING & RAG FUNCTIONS
# ----------------------
def get_embedding(text):
    response = open_ai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def get_top_k(user_question: str, k: int = 5):
    q_emb = get_embedding(user_question)
    results = index.query(vector=q_emb, top_k=k, include_metadata=True)
    return results.matches if results.matches else []

def build_prompt(question: str, contexts):
    context_text = "\n\n".join(
        [f"[{i+1}] {c['metadata']['text']}" for i, c in enumerate(contexts)]
    )
    prompt = f"""
You are a helpful mental-health FAQ assistant.
Answer the user's question using ONLY the context below.
If the context doesn't contain the answer, say you don't know and recommend seeking professional help.

Context:
{context_text}

User question:
{question}

Answer:
"""
    return prompt

def answer_question(user_question, top_k=5, model="gpt-3.5-turbo"):
    top_chunks = get_top_k(user_question, k=top_k)
    prompt = build_prompt(user_question, top_chunks)

    response = open_ai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful mental-health FAQ assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ----------------------
# ROUTES
# ----------------------
@app.route("/", methods=["GET"])
def serve_index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.json
        q_request = QuestionRequest(**data)
        ans = answer_question(q_request.question, top_k=q_request.top_k)
        return jsonify({"answer": ans})
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------
# RUN APP
# ----------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

