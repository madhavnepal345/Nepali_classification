from sklearn.metrics import accuracy_score,precision_socre, recall_score, f1_score,f1_score

def evaluate(y_true,y_pred):
    return{
        "accuracy":accuracy_score(y_true,y_pred),
        "precision":precision_socre(y_true,y_pred),
        "recall":recall_score(y_true,y_pred),
        "f1_score":f1_score(y_true,y_pred)
    }