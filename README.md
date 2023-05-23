# ALCE


## Running Experiments

```
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

## Evaluation

### ASQA

```
python eval.py --f {path/to/result/file} --citations --qa --mauve
```

### QAMPARI

```
python eval.py --f {path/to/result/file} --citations
```

### ELI5

```
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve
```
