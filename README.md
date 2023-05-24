# Enabling Large Language Models to Generate Text with Citations

This repository contains the code and data for paper "Enabling Large Language Models to Generate Text with Citations". 
In this paper, we propose ALCE, a benchmark for **A**utomatic **L**LMs' **C**itation Evaluation. 
ALCE contains three datasets: ASQA, QAMPARI, and ELI5.
We provide automatic evaluation code of LLM generations around three dimensions: fluency, correctness, and citation quality. 
This repository also includes code to reproduce the baselines in our paper.

:bulb: **ALCE** is the latin word for moose.

**************************** **Updates** ****************************

* 5/23: We released the first version of our code. Use at caution! We are still working on cleaning the code to make it more easy to use and the API may change anytime.


## Quick Links

  - [Requirements](#requirements)
  - [Data](#data)
  - [Code Structure](#code-structure)
  - [Reproducing Baselines](#reproducing-baselines)
  - [Evaluation](#evaluation)
  - [Bug or Questions](#bug-or-questions)
  - [Citation](#citatin)


## Requirements

Please install the latest versions of PyTorch (`torch`), Huggingface Transformers (`transformers`), and the OpenAI API package (`openai`).

## Code Structure

* `run.py`: run file to reproduce our baseline generations.
* `eval.py`: eval file to evaluate generations.
* `prompts`: folder that contains all prompt files.
* `configs/`: folder that contains all config files to reproduce baselines.
* `tools/`: misc code (generate summaries/snippets, reranking, etc.)

## Data

You can download datasets (along with retrieval results) by running the following command:

```bash
bash download_data.sh
```

All the data will be stored in `data/`.


## Reproducing Baselines


You can reproduce baselines from our paper by 

```bash
python run.py --config configs/{config_name}
```

You can also overwrite any arguments in the config file or add new arguments simply through command line:
```
python run.py --config configs/{config_name} --seed 43 --model vicuna-13b
```

* For OpenAI models (for example, ChatGPT), you need to set the environment variable `OPENAI_API_KEY` and `OPENAI_ORG_ID`. If you are using the Azure OpenAI API, you need to set the enviroment variable of `OPENAI_API_KEY` and `OPENAI_API_BASE`. You also need to add the flag `--azure`. 
    * Note that in Azure OpenAI API, ChatGPT's name is different and you should set it by `--model gpt-35-turbo`. 
* For the open-source models, you also need to set the environment variable `LLAMA_ROOT` to the directory containing the weights folder for the model. 
For example, you should be able to load the LLaMA-13B with the following line:
```
import os
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(os.path.join(os.environ["LLAMA_ROOT"], "llama-13b"))
```

(We are still working on cleaning the code and adding more detailed README.)

## Evaluation

(We are still working on cleaning the code and adding more detailed README. )

### ASQA

```bash
python eval.py --f {path/to/result/file} --citations --qa --mauve
```

### QAMPARI

```bash
python eval.py --f {path/to/result/file} --citations
```

### ELI5

```bash
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve
```

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