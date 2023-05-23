import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from nltk import sent_tokenize
import re
import numpy as np
import string
import torch
from searcher import SearcherWithinDocs

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Output data file")
    parser.add_argument("--retriever", type=str, default="gtr-t5-large", help="Retriever to use. Options: `tfidf`, `gtr-t5-large`")
    parser.add_argument("--retriever_device", type=str, default="cuda", help="Where to put the dense retriever if using. Options: `cuda`, `cpu`")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing citations")
    parser.add_argument("--external_docs", type=str, default=None, help="Use external documents")
    
    args = parser.parse_args()

    data = json.load(open(args.f))
    new_data = []
    if args.external_docs is not None:
        external = json.load(open(args.external_docs))
    
    # Load retrieval model
    if "gtr" in args.retriever:
        from sentence_transformers import SentenceTransformer
        gtr_model = SentenceTransformer(f'sentence-transformers/{args.retriever}', device=args.retriever_device)
    
    for idx, item in enumerate(tqdm(data['data'])):
        doc_list = item['docs']
        if args.external_docs is not None:
            assert external[idx]['question'] == item['question']
            doc_list = external[idx]['docs']
        searcher = SearcherWithinDocs(doc_list, args.retriever, model=gtr_model, device=args.retriever_device)
        
        output = item["output"].strip().split("\n")[0] # Remove new lines and content after
        output = item["output"].replace("<|im_end|>", "")
        if "qampari" in args.f:
            sents = [item['question'] + ' ' + x.strip() for x in item['output'].rstrip(".").split(",")]
        else:
            sents = sent_tokenize(output)
    
        new_output = ""
        for sent in sents:
            original_ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] 

            if len(original_ref) == 0 or args.overwrite:
                print("\n-----")
                print("Original sentence:", sent)
                print("Original ref:", original_ref)
                sent = remove_citations(sent)
                best_doc_id = searcher.search(sent)
                print("New ref:", best_doc_id)
                sent = f"[{best_doc_id+1}] " + sent
                print("New sentence:", sent)
                if "qampari" in args.f:
                    new_output += sent.replace(item['question'], '').strip() + ", "
                else:
                    new_output += sent + " "
            else:
                if "qampari" in args.f:
                    new_output += sent.replace(item['question'], '').strip() + ", "
                else:
                    new_output += sent + " "
   
        item['output'] = new_output.rstrip().rstrip(",")
        print("Final output: " + item['output'])
        item['docs'] = doc_list
        new_data.append(item)

    data['data'] = new_data 
    tag = f".{args.retriever}" 
    if args.overwrite:
        tag += "-overwrite"
    if args.external_docs is not None:
        tag += "-external"

    json.dump(data, open(args.f + f".post_hoc_cite{tag}", 'w'), indent=4)

if __name__ == "__main__":
    main()
