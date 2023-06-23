# Enabling Large Language Models to Generate Text with Citations

<p align="center"><img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/moose.png?raw=true" alt="ALCE" width="10%"></p>


This repository contains the code and data for paper [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627). 
In this paper, we propose ALCE, a benchmark for **A**utomatic **L**LMs' **C**itation Evaluation. 
ALCE contains three datasets: ASQA, QAMPARI, and ELI5.
We provide automatic evaluation code of LLM generations around three dimensions: fluency, correctness, and citation quality. 
This repository also includes code to reproduce the baselines in our paper.


:bulb: **ALCE** is the latin word for elk (Europe) or moose (North America).


<img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/ALCE.png?raw=true" alt="ALCE" width="100%">




## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Code Structure](#code-structure)
  - [Reproducing Baselines](#reproducing-baselines)
  - [Evaluation](#evaluation)
  - [Bug or Questions](#bug-or-questions)
  - [Citation](#citatin)


## Requirements

Please install the latest versions of PyTorch (`torch`), HuggingFace Transformers (`transformers`), HuggingFace Accelerate (`accelerate`), and the OpenAI API package (`openai`). This codebase is tested on 
`torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, `accelerate==0.17.1`, and `openai==0.27.4` with Python 3.9.7.

## Data

You can download datasets (along with retrieval results) by running the following command:

```bash
bash download_data.sh
```

All the data will be stored in `data/`. Our data included top-100 DPR/GTR retrieved results for ASQA and QAMPARI, and top-100 BM25 retrieved results for QAMPARI. We also provide reranked oracle retrieval results, where top-5 passages can achieve the same recall as the original top-100 recall.

### Retrieval

You can reproduce the passage retrieval step with the following command:
```bash
python retrieval.py --data {path/to/data} --retriever {bm25/gtr} --output_file {path/to/output}
```

There are additional packages required for the retrieval steps.
Specifically, you need to install `pyserini==0.21.0`(their github [repo](https://github.com/castorini/pyserini/tree/master) is helpful) and `sentence-transformers==2.2.2`.

For the BM25 retrieval over Common Crawl using Sphere, you must first download the index from the Sphere [repo](https://github.com/facebookresearch/Sphere), and set the environmental variable `BM25_SPHERE_PATH` to the path of the downloaded index.
It's important to note that given the large size of the corpus, this step is extremely expensive and time-consuming. We found that larger CPU memory tends to help with the speed. 

For GTR, we first build an index using the DPR wikipedia snapshot, which you can obtain using the download script from the DPR [repo](https://github.com/facebookresearch/DPR), and then setting the environmental variable `DPR_WIKI_TSV` to the path of the tsv file.
Building the dense index can be expensive for GPU memory (we use 80GB GPUs for this) and time-consuming; the entire index will take about 31GB.
If you find this step to be too expensive, please contact us and we can help you with obtaining the index.

To reproduce the DPR retrieval, we refer the DPR [repo](https://github.com/facebookresearch/DPR), which we used the original DPR checkpoint trained on NQ.

## Code Structure

* `run.py`: run file to reproduce our baseline generations.
* `eval.py`: eval file to evaluate generations.
* `prompts`: folder that contains all prompt files.
* `configs/`: folder that contains all config files to reproduce baselines.
* `tools/`: misc code (generate summaries/snippets, reranking, etc.)




## Reproducing Baselines


You can reproduce baselines from our paper by 

```bash
python run.py --config configs/{config_name}
```

You can also overwrite any arguments in the config file or add new arguments simply through command line:
```
python run.py --config configs/{config_name} --seed 43 --model vicuna-13b
```

For the open-source models, you also need to set the environment variable `LLAMA_ROOT` to the directory containing the weights folder for the model. 
For example, you should be able to load the LLaMA-13B with the following line:
```
import os
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(os.path.join(os.environ["LLAMA_ROOT"], "llama-13b"))
```

* For OpenAI models (for example, ChatGPT), you need to set the environment variable `OPENAI_API_KEY` and `OPENAI_ORG_ID`. If you are using the Azure OpenAI API, you need to set the environment variable of `OPENAI_API_KEY` and `OPENAI_API_BASE`. You also need to add the flag `--azure`. 
    * Note that in Azure OpenAI API, ChatGPT's name is different and you should set it by `--model gpt-35-turbo`. 
* For the open-source models, you also need to set the environment variable `LLAMA_ROOT` to the directory containing the weights folder for the model (as LLaMA checkpoints are not available on HuggingFace model hub and can only be loaded locally).

For detailed argument usage, please refer to `run.py`.

Model output along with gold answers and run configs will be stored in a json file in `result/`.


### Post-hoc citation

For closed-book models, one can use `post_hoc_cite.py` to add citations in a post-hoc manner (using GTR-large). To run post-hoc citation, execute
```bash
python post_hoc_cite.py --f result/{RESULT JSON FILE NAME} --external_docs data/{CORRESPONDING DATA}
```

The output file with post-hoc citations will be stored in `result/`, with a suffix `post_hoc_cite.gtr-t5-large-external`.

## Evaluation

ACLE evaluation is implemented in `eval.py`. 

For ASQA, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --qa --mauve
```

For QAMPARI, use the following command
```bash
python eval.py --f {path/to/result/file} --citations
```

For ELI5, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve
```

The evaluation result will be saved in `result/`, with the same name as the input and a suffix `.score`.

## Bug or Questions?

If you have any questions related to the code or the paper, feel free to email Tianyu (`tianyug@cs.princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please cite our paper if you use ALCE in your work:

```bibtex
@article{gao2023enabling,
   title={Enabling Large Language Models to Generate Text with Citations},
   author={Gao, Tianyu and Yen, Howard and Yu, Jiatong and Chen, Danqi},
   year={2023}
}
```