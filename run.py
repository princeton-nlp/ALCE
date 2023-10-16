import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
import openai
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import string
import numpy as np
import re
from searcher import SearcherWithinDocs
import yaml
from utils import *
from nltk import sent_tokenize

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

class LLM:

    def __init__(self, args):
        self.args = args

        if args.openai_api:
            import openai 
            OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
            OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID")
            OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

            if args.azure:
                openai.api_key = OPENAI_API_KEY
                openai.api_base = OPENAI_API_BASE
                openai.api_type = 'azure'
                openai.api_version = '2023-05-15' 
            else: 
                openai.api_key = OPENAI_API_KEY
                openai.organization = OPENAI_ORG_ID

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False) # TODO: For ChatGPT we should use a different one
            # To keep track of how much the API costs
            self.prompt_tokens = 0
            self.completion_tokens = 0
        else:
            self.model, self.tokenizer = load_model(args.model)
        
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0


    def generate(self, prompt, max_tokens, stop=None):
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        if args.openai_api:
            use_chat_api = ("turbo" in args.model and not args.azure) or ("gpt-4" in args.model and args.azure)
            if use_chat_api:
                # For chat API, we need to convert text prompts to chat prompts
                prompt = [
                    {'role': 'system', 'content': "You are a helpful assistant that answers the following questions with proper citations."},
                    {'role': 'user', 'content': prompt}
                ]
            if args.azure:
                deploy_name = args.model

            if use_chat_api:
                is_ok = False
                retry_count = 0
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.ChatCompletion.create(
                            engine=deploy_name if args.azure else None,
                            model=args.model,
                            messages=prompt,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            stop=stop,
                            top_p=args.top_p,
                        )
                        is_ok = True
                    except Exception as error:
                        if retry_count <= 5:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            continue
                        print(error)
                        import pdb; pdb.set_trace()
                self.prompt_tokens += response['usage']['prompt_tokens']
                self.completion_tokens += response['usage']['completion_tokens']
                return response['choices'][0]['message']['content']
            else:
                is_ok = False
                retry_count = 0
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.Completion.create(
                            engine=deploy_name if args.azure else None,
                            model=args.model,
                            prompt=prompt,
                            temperature=args.temperature,
                            max_tokens=max_tokens,
                            top_p=args.top_p,
                            stop=["\n", "\n\n"] + (stop if stop is not None else [])
                        )    
                        is_ok = True
                    except Exception as error:
                        if retry_count <= 5:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            if "triggering Azure OpenAI’s content management policy" in str(error):
                                # filtered by Azure 
                                self.azure_filter_fail += 1
                                return ""
                            continue
                        print(error)
                        import pdb; pdb.set_trace()
                self.prompt_tokens += response['usage']['prompt_tokens']
                self.completion_tokens += response['usage']['completion_tokens']
                return response['choices'][0]['text']
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            stop = [] if stop is None else stop
            stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
            stop_token_ids = list(set([self.tokenizer._convert_token_to_id(stop_token) for stop_token in stop] + [self.model.config.eos_token_id]))
            if "llama" in args.model.lower():
                stop_token_ids.remove(self.tokenizer.unk_token_id)
            outputs = self.model.generate(
                **inputs,
                do_sample=True, temperature=args.temperature, top_p=args.top_p, 
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                eos_token_id=stop_token_ids
            )
            generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    # Prompt file is a json file that contains the following fields:
    # - instruction: the instruction, which will appear at the beginning of each demo and the test example
    # - demo_sep: the separator between each demo, for example, "\n\n\n"
    # - demo_prompt: the prompt for the demo, for example, "Instruction: {INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
    #     - {INST}: the instruction
    #     - {D}: the documents
    #     - {Q}: the question
    #     - {A}: the answers
    # - doc_prompt, the prompt for each document, for example, "Document [{ID}](Title: {T}): {P}", where
    #     - {ID}: the document id, staring from 1
    #     - {T}: the document title
    #     - {P}: the document text
    # - demos: a list of demo examples, each of which should have
    #     - question: the question
    #     - docs: the documents ("title" and "text")
    #     - answer: the answer to show in the demo. If it is a list, they will be concatenated by "\n". This is useful when the answer includes interactive components.
    # Note that this python file will sample `--shot` demos from the prompt file given the random seed `--seed`
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")

    # Evaluation file is a json file that contains a list of item, each of which contains
    # - question: the question
    # - answer: the answer
    # - docs: the documents, each of which contains "title", "text"
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="Whether to remove the documents in the demos")
    parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")

    # Interactive
    parser.add_argument("--interactive", type=bool, default=False, help="Whether to run in interactive mode")
    parser.add_argument("--interactive_query", type=str, default=None, help="The query to use in interactive mode, either `doc_id` (corresponding to interact in paper) or `search` (corresponding to inlinesearch in paper).")
    parser.add_argument("--retriever", type=str, default=None, help="When using interactive search mode, which retriever to use. Options: `tfidf`, `gtr-t5-large`")
    parser.add_argument("--retriever_device", type=str, default="cuda", help="Where to put the dense retriever if using. Options: `cuda`, `cpu`")
    parser.add_argument("--retrieve_in_all_docs", type=bool, default=False, help="Retrieve in all documents instead of just top ndoc")
    parser.add_argument("--max_turn", type=int, default=10, help="Max number of all actions")
    parser.add_argument("--max_doc_show", type=int, default=3, help="Max number of documents to show at one time.")
    parser.add_argument("--force_cite_show", type=bool, default=False, help="Force citing the documents that are shown to the model")


    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    if "turbo" in args.model:
        # ChatGPT has a longer max length
        args.max_length = 4096

    if "16k" in args.model:
        args.max_length = 16384
    elif "32k" in args.model:
        args.max_length = 32768
    elif "turbo" in args.model:
        args.max_length = 4096
    elif "gpt-4" in args.model:
        args.max_length = 8192
    elif "llama-2" in args.model.lower() or "llama2" in args.model.lower():
        args.max_length = 4096


    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")
        

    # Load the model or setup the API
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file))

    # Generate the demonstration part
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        ndoc = args.ndoc
        if args.no_doc_in_demo:
            ndoc = 0
        elif args.fewer_doc_in_demo:
            assert args.ndoc_in_demo is not None
            ndoc = args.ndoc_in_demo
        head_prompt += make_demo(
            train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter 
        )
        head_prompt += prompt_data["demo_sep"]

    # Sample quick test
    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):
        eval_data[idx]['prompt'] = head_prompt + make_demo(
            eval_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter, 
            test=True
        )
        doc_list = get_shorter_text(eval_item, eval_item["docs"], args.ndoc, args.use_shorter) if args.use_shorter is not None else eval_item["docs"][:args.ndoc]
        if not args.retrieve_in_all_docs:
            # If --retrieve_in_all_docs, we keep the original docs and do not trim them by ndoc
            # Otherwise, take the new docs (truncated by ndoc and filtered if using summary/extraction)
            eval_data[idx]['docs'] = doc_list
        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")

    # Load retriever for interactive search 
    if args.interactive and args.interactive_query == "search" and "gtr" in args.retriever:
        from sentence_transformers import SentenceTransformer
        gtr_model = SentenceTransformer(f'sentence-transformers/{args.retriever}', device=args.retriever_device)
        from searcher import SearcherWithinDocs

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            if args.interactive:
                print("============ Interactive =============")
                output_answer = ""
                doc_list = item['docs']

                interactive_prompt = prompt.rstrip() + "\n" # Start a new line
                inline_doc = ""
                num_turn = 0
                
                doc_history = []
                while True:
                    # For each action, it should end at the new line
                    # Three possible actions
                    # - Check: Document [1][2][3] / search query
                    # - Output: output 
                    # - End
                    num_turn += 1
                    new_prompt = interactive_prompt + inline_doc
                    new_prompt_len = len(llm.tokenizer.tokenize(new_prompt))

                    if idx == 0:
                        print(f"-------------- Step {num_turn} prompt --------------")
                        print(new_prompt)
                        print("-----------------------------")

                    output = llm.generate(new_prompt, min(args.max_new_tokens, args.max_length-new_prompt_len), stop=["\n", "\n\n"])

                    if len(inline_doc) > 0:
                        output = "Output: " + output # "Output: " was included in inline_doc
                    inline_doc = "" # Delete inline_doc after use
                    interactive_prompt += output + "\n"
                    logger.info(f"Model output: \"{output}\"")

                    if output.strip().lower()[:3] == "end":
                        # Model decides to end the generation
                        break
                    elif "sorry" in output.lower() and ("relevant document" in output.lower() or "relevant information" in output.lower()) or "none of the documents" in output.lower():
                        # Instruction-tuned model may abstain from answer the question
                        break
                    elif output.strip().lower()[:5] == "check" or output.strip().lower()[:6] == "search":
                        # Checkout or search documents
                        if args.interactive_query == "search":
                            query = output.replace("Search:", "").replace("search:", "").strip()
                            if len(doc_list) == 0:
                                show_doc_ids = []
                            else:
                                searcher = SearcherWithinDocs(doc_list, args.retriever, model=gtr_model, device=args.retriever_device)
                                show_doc_ids = [int(searcher.search(query))]
                        elif args.interactive_query == "doc_id":
                            show_doc_ids = [int(r[1:])-1 for r in re.findall(r"\[\d+", output)] # In text citation id starts from 1
                            show_doc_ids = [doc_id for doc_id in show_doc_ids if doc_id < len(doc_list) and doc_id >= 0]
                            show_doc_ids = show_doc_ids[:args.max_doc_show] # Avoiding showing too many documents
                        else:
                            raise NotImplementedError

                        inline_doc = "".join([make_doc_prompt(doc_list[doc_id], doc_id, prompt_data["doc_prompt"]) for doc_id in show_doc_ids])
                        inline_doc += "Output:" # Force the model to generate output in the next step
                        doc_history.append(show_doc_ids)
                    elif output.strip().lower()[:6] == "output":
                        output = output.strip().replace("Output:", "").strip()
                        if args.force_cite_show:
                            output = remove_citations(output)
                            if len(doc_history) == 0:
                                logger.warn("No doc history??")
                            else:
                                # Just cite whatever documents the model has seen in the last step
                                if "qampari" in args.eval_file:
                                    output = ", ".join(["".join([f"[{doc+1}]" for doc in doc_history[-1]]) + " " + entity.strip() for entity in output.rstrip().rstrip(",").split(",")]) + ", "
                                else:
                                    output = " ".join(["".join([f"[{doc+1}]" for doc in doc_history[-1]]) + " " + o for o in sent_tokenize(output)]) + "."
                        output_answer += " " + output 
                    else:
                        # Sometimes model starts to output random things.
                        break
                    
                    if num_turn >= args.max_turn:
                        logger.warning("Reach maximum number of turns. Terminate now.")
                        break
                
                if "qampari" in args.eval_file:
                    output_answer = output_answer.rstrip().rstrip(",")
                output_array.append(output_answer)
                item['prompt'] = interactive_prompt
                item['doc_history'] = doc_history
            else: 
                output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
                item['prompt'] = prompt
            
            output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            if output_array[-1].endswith("End."):
                output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Prompt length={prompt_len}")
            logger.info(f"Question: {item['question']}")
            logger.info(f"Gold answer: {item['answer']}")
            logger.info(f"Final model output: {output_array[-1]}") 
        
        item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.tag}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}"
    if args.azure:
        name += "-azure"
    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"
    if args.no_doc_in_demo:
        name += "-no_doc_in_demo"
    if args.fewer_doc_in_demo:
        name += f"-{args.ndoc_in_demo}_doc_in_demo"
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
    if args.force_cite_show:
        name += f"-forceciteshow"

    
    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }
    if args.openai_api:
        logger.info(f"Token used: prompt {llm.prompt_tokens}; completion {llm.completion_tokens}")
        if "turbo" in args.model:
            p_price, c_price = 0.0015, 0.002
            if "16k" in args.model:
                p_price, c_price = 0.003, 0.004
        elif "gpt4" in args.model or "gpt-4" in args.model:
            p_price, c_price = 0.03, 0.06
            if "32k" in args.model:
                p_price, c_price = 0.06, 0.12
        else:
            logger.warn("Cannot find model price")
            p_price, c_price = 0, 0

        eval_data["total_cost"] = llm.prompt_tokens / 1000 * p_price + llm.completion_tokens / 1000 * c_price        

        logger.info(f"Unit price (Oct 16, 2023, prompt/completion): {p_price}/{c_price}")
        logger.info(f"Total cost: %.1f" % (eval_data["total_cost"]))

        if args.azure:
            eval_data["azure_filter_fail"] = llm.azure_filter_fail 

    if not os.path.exists("result"):
        os.makedirs("result")
    json.dump(eval_data, open("result/" + name + ".json", "w"), indent=4)

if __name__ == "__main__":
    main()
