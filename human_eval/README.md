# Human Evaluation

In this directory, you can find the human evaluation results in two files: `human_eval_utility_completed.json` and `human_eval_citations_completed.json`. 
We evaluated a sample of ASQA and ELI5 results from select models. For more details, please refer to the paper. 

### Utility
In `human_eval_utility_completed.json`, each model linked to a dictionary where the key is the question id and the value is the model output. This is identical to the data you would find after running evaluation with one additional field -- `utility_score` is rated from 1-5.

### Citations
In `human_eval_citations_completed.json`, each model linked to a dictionary where the key is the question id and the value is the model output. This is identical to the data you would find after running evaluation with a few additional fields:

- `citation_precision_score` can be found for every valid citation in every sentence. The instruction given to the annotator is: "Given the sentence and one of its cited documents, please rate if the document fully supports all claims in the sentence (2), the document partially supports the claims in the sentence (1), or if the document does not support any claims made in the sentence (0)."
- `sentence_recall_score` can be found for every sentence. The instruction given to the annotator is: "Given the sentence and its cited documents, please rate if the model response is fully supported by the documents (1) or if the model response is not fully supported by the documents (0). If the response is fully supported, that means all factually claims made by the response are found in and supported by at least one of the documents. Otherwise, the response is not fully supported."

We then calculate the humans `overall_precision_score` and `overall_recall_score`. Furthermore, we included the automatic evaluation results for ease of evaluation.
