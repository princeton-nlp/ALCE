import argparse
import openai
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np

OPENAI_API_KEY = ""
OPENAI_ORG_ID = ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Data file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301", help="What model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for decoding")
    parser.add_argument("--target", type=str, default="summary", help="Summary or extraction? Options: `summary`, `extraction`")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--ndoc", type=int, default=20, help="Generate summary for the top-ndoc documents")
    args = parser.parse_args()

    openai.organization = OPENAI_ORG_ID
    openai.api_key = OPENAI_API_KEY

    data = json.load(open(args.f))
    total_tokens = 0
    new_f_temp = args.f.replace(".json", f"_w_{args.target}_top{args.ndoc}_workinprogress.json")

    for item_id, item in enumerate(tqdm(data)):
        for doc_id, doc in enumerate(item['docs'][:args.ndoc]):
            if args.target in doc:
                print("pass")
                continue
            if args.target == "summary":
                prompt = [
                    {'role': 'system', 'content': "You are a helpful assistant that summarizes the following documents with respect to questions of interest."},
                    {'role': 'user', 'content': f"Summarize the following document within 50 words with the question of interest \"{item['question']}\" Return \"irrelevant\" if the document is irrelevant to the question. Try to keep all the important dates, numbers, and names.\n\nTitle: {doc['title']}\nText: {doc['text']}\nSummary:"}
                ]
            elif args.target == "extraction":
                prompt = [
                    {'role': 'system', 'content': "You are a helpful assistant that extracts answers to questions from given documents."},
                    {'role': 'user', 'content': f"Given the follow passage and the question \"{item['question']}\", extract a useful span from the passage that can answer the question. Resolve all the coreference issues to make the extracted span understandable standalone. If the passage is not helpful for answering the question, return \"irrelevant\".\n\nTitle: {doc['title']}\nText: {doc['text']}\nExtracted span:"}
                ]
            else:
                raise NotImplementedError

            ok = False
            retry_count = 0
            while not ok:
                retry_count += 1
                try:
                    response = openai.ChatCompletion.create(
                        model=args.model,
                        messages=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    ok = True
                except Exception as error:
                    if retry_count <= 5:
                        print(f"Retry for {retry_count} times ({error})")
                        continue
                    print(error)
                    import pdb; pdb.set_trace()
            
            content = response['choices'][0]['message']['content'].strip()
            total_tokens += response['usage']['total_tokens']
            data[item_id]['docs'][doc_id][args.target] = content

            print("--------------------")
            print(f"Question: {item['question']}")
            print(f"Document ({doc['title']}): {doc['text']}")
            print("---")
            print(f"{args.target}: {content}")
        
        # Save intermediate results in case the program crashes
        if item_id % 10 == 0:
            json.dump(data, open(new_f_temp, "w"), indent=4)
        
    new_f = args.f.replace(".json", f"_w_{args.target}_top{args.ndoc}.json")
    json.dump(data, open(new_f, "w"), indent=4)

    print("Cost: %.1f" % (total_tokens / 1000 * 0.002))

if __name__ == "__main__":
    main()
