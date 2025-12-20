import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(model,class_label,top_n=20):
    tfidf=model.named_steps["tfidf"]
    clf=model.named_steps["clf"]

    feature_names=np.array(tfidf.get_feature_names_out())
    class_index=list(clf.classes_).index(class_label)
    coefs=clf.coef_[class_index]
    top_indices=np.argsort(coefs)[-top_n:]
    plt.figure(figsize=(10,6))
    plt.barh(feature_names[top_indices],coef[top_indices])