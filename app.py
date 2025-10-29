# app.py
import os, json, re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder  # optional; if not installed, code falls back
import faiss
import openai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")
openai.api_key = OPENAI_KEY

INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "metadata.jsonl")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 20  # how many to fetch for reranking

app = FastAPI(title="Hack-A-Cure RAG API")

class QueryRequest(BaseModel):
    query: str
    top_k: int

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

# load models + index
if not (Path(INDEX_FILE).exists() and Path(META_FILE).exists()):
    raise RuntimeError("Missing index or metadata. Run the build script first.")

embed_model = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_FILE)
# load metadata into a list (only text + provenance needed)
metadata = [json.loads(line) for line in open(META_FILE, encoding="utf-8")]

# try to load reranker
use_reranker = False
try:
    reranker = CrossEncoder(RERANKER_MODEL)
    use_reranker = True
except Exception:
    reranker = None
    use_reranker = False

# utility regex for numeric guard
NUM_RE = re.compile(r'\d+(?:\.\d+)?\s*(?:mg/kg|mg|g|ml|IU|%|/day|/kg/day|q\d+h|daily|per day|per week)?', re.IGNORECASE)

def numeric_tokens_in_text(s):
    return NUM_RE.findall(s or "")

def numeric_guard(answer, returned_texts):
    nums = numeric_tokens_in_text(answer)
    if not nums:
        return True
    joined = " ".join(returned_texts)
    for n in nums:
        if re.search(re.escape(n), joined, flags=re.IGNORECASE):
            return True
        if re.search(re.escape(n.replace('-', '–')), joined, flags=re.IGNORECASE):
            return True
    return False

def synthesize_answer(question, contexts):
    ctx_text = "\n\n".join(f"[{i}] {c}" for i,c in enumerate(contexts))
    prompt = (
        "You are a medical assistant. Use ONLY the following context snippets to answer the question.\n"
        "If the snippets do not contain the answer, reply exactly: Insufficient information in dataset.\n"
        "Answer in 1-2 concise sentences and do not add extra commentary.\n\n"
        f"CONTEXTS:\n{ctx_text}\n\nQUESTION: {question}\n\nANSWER:"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are a precise, evidence-based assistant."},
                  {"role":"user","content":prompt}],
        temperature=0,
        max_tokens=300
    )
    return resp["choices"][0]["message"]["content"].strip()

@app.post("/query", response_model=QueryResponse)
def query_route(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query is required")
    top_k = max(0, int(req.top_k))
    # embed query
    q_emb = embed_model.encode([req.query], convert_to_numpy=True)
    # normalize and search
    faiss.normalize_L2(q_emb)
    candidate_k = max(CANDIDATE_K, top_k*3 if top_k>0 else CANDIDATE_K)
    D, I = index.search(q_emb, candidate_k)
    idxs = [int(i) for i in I[0] if i != -1]
    # map to texts and metadata
    candidates = [metadata[i] for i in idxs]
    candidate_texts = [c.get("text","") for c in candidates]
    # rerank if available
    if use_reranker and candidate_texts:
        scores = reranker.predict([(req.query, t) for t in candidate_texts])
        ranked = sorted(zip(scores, candidates, candidate_texts), key=lambda x: -x[0])
        ranked_texts = [r[2] for r in ranked]
        ranked_metas = [r[1] for r in ranked]
    else:
        ranked_texts = candidate_texts
        ranked_metas = candidates
    # choose top_k contexts to RETURN to evaluator (plain strings)
    returned_texts = ranked_texts[:top_k] if top_k>0 else []
    # Synthesize using ONLY returned_texts (this matches evaluator expectation)
    answer = synthesize_answer(req.query, returned_texts)
    # numeric guard
    if answer.lower().strip() != "insufficient information in dataset.":
        if not numeric_guard(answer, returned_texts):
            answer = "Insufficient information in dataset."
    return {"answer": answer, "contexts": returned_texts}