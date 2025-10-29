# scripts/build_index_from_combined.py
import os, json, gc
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# CONFIG
COMBINED = "data/final_combined_chunks.jsonl"
OUT_DIR = "data/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # CPU-friendly; swap for bio model if you have one
BATCH_SIZE = 64  # lower to 16 if memory is tight

def ensure_out():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def gen_chunks():
    with open(COMBINED, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)

def main():
    ensure_out()
    model = SentenceTransformer(EMBED_MODEL)
    meta_out_path = os.path.join(OUT_DIR, "metadata.jsonl")
    # If re-run, we will overwrite index and metadata to ensure consistency
    if os.path.exists(meta_out_path):
        os.remove(meta_out_path)
    vectors = []
    metas = []
    index = None
    total = 0
    for rec in gen_chunks():
        vectors.append(rec.get("text",""))
        metas.append(rec)
        if len(vectors) >= BATCH_SIZE:
            embs = model.encode(vectors, convert_to_numpy=True, show_progress_bar=False)
            if index is None:
                dim = embs.shape[1]
                index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embs)
            index.add(embs)
            # write metadata
            with open(meta_out_path, "a", encoding="utf-8") as mm:
                for m in metas:
                    mm.write(json.dumps(m, ensure_ascii=False) + "\n")
            total += len(metas)
            print("Indexed:", total)
            vectors = []; metas = []
            gc.collect()
    # leftover
    if vectors:
        embs = model.encode(vectors, convert_to_numpy=True, show_progress_bar=False)
        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embs)
        index.add(embs)
        with open(meta_out_path, "a", encoding="utf-8") as mm:
            for m in metas:
                mm.write(json.dumps(m, ensure_ascii=False) + "\n")
        total += len(metas)
        print("Final indexed total:", total)
    # Save index
    if index is not None:
        faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
        print("FAISS index and metadata saved to", OUT_DIR)
    else:
        print("No data indexed. Check combined file path.")

if __name__ == "__main__":
    main()