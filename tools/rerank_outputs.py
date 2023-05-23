import argparse
import json
import re
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from nltk import sent_tokenize
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def rerank_outputs(data, mode, at_most_citations=None, qampari=False):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    def _nli_prob(passage, claim):
        input_text =f"premise: {passage} hypothesis: {claim}"
        input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
        with torch.inference_mode():
            outputs = autoais_model.generate(input_ids, output_scores=True, return_dict_in_generate=True)
            outputs = outputs.scores[0]
        one_input_id = 209
        prob = torch.nn.functional.softmax(outputs[0], -1)[one_input_id].item()
        return prob if "prob" in mode else 1 if prob >=0.5 else 0

    logger.info(f"Reranking outputs using mode {mode}...")

    for i, item in enumerate(tqdm(data)):
        outputs = item["output"]
        assert type(outputs) == list

        best_score = 0
        best_output = ""

        for output in outputs:
            if qampari:
                sents = [item["question"] + " " + x.strip() for x in output.rstrip(".").split(",")]
            else:
                sents = sent_tokenize(output)
            if len(sents) == 0:
                continue
            target_sents = [remove_citations(sent).strip() for sent in sents]

            entail = 0

            for sent_idx, sent in enumerate(sents):
                target_sent = target_sents[sent_idx]
                ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)]
                if len(ref) == 0:
                    flag = 0
                elif any([ref_id >= len(item["docs"]) for ref_id in ref]):
                    flag = 0
                else:
                    if at_most_citations is not None:
                        ref = ref[:at_most_citations]
                    passage = "\n".join([f"Title: {doc['title']}\n{doc['text']}" for doc in item["docs"]])
                    flag = _nli_prob(passage, target_sent)
                entail += flag
            score = entail / len(sents)

            if score > best_score:
                best_score = score
                best_output = output

        item["output"] = best_output

    logger.info(f"Done with reranking outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    parser.add_argument("--rerank_mode", type=str, default=None, help="How to rerank outputs: {prob, discrete}")
    parser.add_argument("--at_most_citations", type=int, default=3, help="Max num citations to consider.")
    args = parser.parse_args()

    with open(args.f) as f:
        data_with_config = json.load(f)
    data = data_with_config['data']
    qampari = "qampari" in args.f

    assert args.rerank_mode is not None

    rerank_outputs(data, mode=args.rerank_mode, at_most_citations=args.at_most_citations, qampari=qampari)
    with open(args.f + ".rerank", "w") as f:
        json.dump(data_with_config, f, indent=4)


if __name__ == "__main__":
    main()