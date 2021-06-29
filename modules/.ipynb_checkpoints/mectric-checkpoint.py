# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precsion, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': round(acc, 4),
        'f1': round(f1,4)
    }