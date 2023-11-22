import argparse
import csv
import json
import os
import time
import pickle

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

TOPK = 100

def bm25_sphere_retrieval(data):
    from pyserini.search import LuceneSearcher
    index_path = os.environ.get("BM25_SPHERE_PATH")
    print("loading bm25 index, this may take a while...")
    searcher = LuceneSearcher(index_path)

    print("running bm25 retrieval...")
    for d in tqdm(data):
        query = d["question"]
        try:
            hits = searcher.search(query, TOPK)
        except Exception as e:
            #https://github.com/castorini/pyserini/blob/1bc0bc11da919c20b4738fccc020eee1704369eb/scripts/kilt/anserini_retriever.py#L100
            if "maxClauseCount" in str(e):
                query = " ".join(query.split())[:950]
                hits = searcher.search(query, TOPK)
            else:
                raise e

        docs = []
        for hit in hits:
            h = json.loads(str(hit.docid).strip())
            docs.append({
                "title": h["title"],
                "text": hit.raw,
                "url": h["url"],
            })
        d["docs"] = docs


def gtr_build_index(encoder, docs):
    with torch.inference_mode():
        embs = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype("float16")

    GTR_EMB = os.environ.get("GTR_EMB")
    with open(GTR_EMB, "wb") as f:
        pickle.dump(embs, f)
    return embs


def gtr_wiki_retrieval(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device = device)

    questions = [d["question"] for d in data]
    with torch.inference_mode():
        queries = encoder.encode(questions, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        queries = torch.tensor(queries, dtype=torch.float16, device="cpu")

    # the wikipedia split from DPR repo: https://github.com/facebookresearch/DPR
    DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
    docs = []
    print("loading wikipedia file...")
    with open(DPR_WIKI_TSV) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            docs.append(row[2] + "\n" + row[1])

    GTR_EMB = os.environ.get("GTR_EMB")
    if not os.path.exists(GTR_EMB):
        print("gtr embeddings not found, building...")
        embs = gtr_build_index(encoder, docs)
    else:
        print("gtr embeddings found, loading...")
        with open(GTR_EMB, "rb") as f:
            embs = pickle.load(f)

    del(encoder) # save gpu mem

    gtr_emb = torch.tensor(embs, dtype=torch.float16, device=device)

    print("running GTR retrieval...")
    for qi, q in enumerate(tqdm(queries)):
        q = q.to(device)
        scores = torch.matmul(gtr_emb, q)
        score, idx = torch.topk(scores, TOPK)
        ret = []
        for i in range(idx.size(0)):
            title, text = docs[idx[i].item()].split("\n")
            ret.append({"id": str(idx[i].item()+1),"title": title, "text": text, "score": score[i].item()})
        data[qi]["docs"] = ret
        q = q.to("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--retriever", type=str, default=None, help="options: bm25/gtr")
    parser.add_argument("--data_file", type=str, default=None, help="path to the data file")
    parser.add_argument("--output_file", type=str, default=None, help="same format as the data file but with the retrieved docs.")
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = json.load(f)

    if args.retriever == "bm25":
        bm25_sphere_retrieval(data)
    elif args.retriever == "gtr":
        gtr_wiki_retrieval(data)
    else:
        raise NotImplementedError

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
