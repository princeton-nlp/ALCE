from sklearn.metrics import confusion_matrix, cohen_kappa_score
import json
import numpy as np

with open("human_eval_citations_completed.json") as f:
    citation_data = json.load(f)
    
# sent human recall, sent human prec, sent auto recall, sent auto prec, cite human prec, cite auto prec
all_scores = [[],[],[],[],[],[]]

print("model,cite recall (human),cite prec (human),cite recall (automatic),cite prec (automatic)\n")
for dataset, models in citation_data.items():
    for model, items in models.items():
        model_scores = [[],[],[],[],[],[]]
        
        for id, item in items.items():
            if id == "overall_results":
                continue
            item_scores = [[],[],[],[],[],[]]
            
            for i, sent in enumerate(item["sentences"]):
                
                # these are all sentence level
                human_recall = sent["sentence_recall_score"]
                human_precision = sent["sentence_precision_score"]
                auto_recall = item["automatic_recall_scores"][i]
                auto_precision = item["automatic_precision_scores"][i]
                
                assert human_recall == 0 or human_recall == 1
                assert human_precision == 0 or human_precision == 1
                assert auto_recall == 0 or auto_recall == 1
                assert auto_precision == 0 or auto_precision == 1
                human_precision = min(human_recall, human_precision)
                
                # these are citation level scores
                human_citation_precision_scores = [x["citation_precision_score"] for x in sent["citations"]]
                # for precision, either 1 or 2 counts as support, so we use min
                human_citation_precision_scores = [0 if human_recall == 0 else min(x, 1) for x in human_citation_precision_scores]
                auto_citation_precision_scores = item["automatic_citation_precision_scores"][i]
                auto_citation_precision_scores = [0 if auto_recall == 0 else min(x, 1) for x in auto_citation_precision_scores]

                assert len(human_citation_precision_scores) == len(auto_citation_precision_scores)
                                
                item_scores[0].append(human_recall)
                item_scores[1].append(human_precision)
                item_scores[2].append(auto_recall)
                item_scores[3].append(auto_precision)
                item_scores[4] += human_citation_precision_scores
                item_scores[5] += auto_citation_precision_scores
        
            for i, s in enumerate(all_scores):
                all_scores[i] += item_scores[i]
                
            item_scores = [x if len(x) > 0 else [0] for x in item_scores]
                
            for i, s in enumerate(model_scores):
                s.append(np.mean(item_scores[i]))
        
        print(f"{model},{np.mean(model_scores[0])*100:.01f},{np.mean(model_scores[4])*100:.01f},{np.mean(model_scores[2])*100:.01f},{np.mean(model_scores[5])*100:.01f}")
    print()
            
print()

print("-----citation recall scores-----")
cm = confusion_matrix(all_scores[0], all_scores[2], labels=[0,1])
print("recall cm (total = {cm.sum()})\n", cm)
tn, fp, fn, tp = cm.ravel()
acc = (tn + tp) / (cm.sum())
precision = tn / (tn+fn)
recall = tn / (tn+fp)
print(f"accuracy, recall, precision = {acc*100:.01f},{recall*100:.01f},{precision*100:.01f}")
print(f"recall cohens kappa between automatic and human evaluation:")
print(cohen_kappa_score(all_scores[0], all_scores[2]))
print()

print("-----citation prec scores-----")
cm = confusion_matrix(all_scores[4], all_scores[5], labels=[0,1])
print(f"precision cm (total = {cm.sum()})\n", cm)
tn, fp, fn, tp = cm.ravel()
acc = (tn + tp) / (cm.sum())
precision = tn / (tn+fn)
recall = tn / (tn+fp)
print(f"accuracy, recall, precision = {acc*100:.01f},{recall*100:.01f},{precision*100:.01f}")
print(f"precision cohens kappa between automatic and human evaluation:")
print(cohen_kappa_score(all_scores[4], all_scores[5]))