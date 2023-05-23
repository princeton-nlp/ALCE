import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from torch.nn.functional import normalize

def doc_to_text_tfidf(doc):
    return doc['title'] + ' ' + doc['text']

def doc_to_text_dense(doc):
    return doc['title'] + '. ' + doc['text']


class SearcherWithinDocs:

    def __init__(self, docs, retriever, model=None, device="cuda"):
        self.retriever = retriever
        self.docs = docs
        self.device = device
        if retriever == "tfidf":
            self.tfidf = TfidfVectorizer()
            self.tfidf_docs = self.tfidf.fit_transform([doc_to_text_tfidf(doc) for doc in docs])
        elif "gtr" in retriever: 
            self.model = model
            self.embeddings = self.model.encode([doc_to_text_dense(doc) for doc in docs], device=self.device, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)
        else:
            raise NotImplementedError

    def search(self, query):
        # Return the top-1 result doc id

        if self.retriever == "tfidf":
            tfidf_query = self.tfidf.transform([query])[0]
            similarities = [cosine_similarity(tfidf_doc, tfidf_query) for tfidf_doc in self.tfidf_docs]
            best_doc_id = np.argmax(similarities)
            return best_doc_id
        elif "gtr" in self.retriever:
            q_embed = self.model.encode([query], device=self.device, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)
            score = torch.matmul(self.embeddings, q_embed.t()).squeeze(1).detach().cpu().numpy()
            best_doc_id = np.argmax(score)
            return best_doc_id
        else:
            raise NotImplementedError
