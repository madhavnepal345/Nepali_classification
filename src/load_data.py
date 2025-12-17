from datasets import load_dataset


def load_data(train_samples=1000,test_samples=500):
    dataset=load_dataset("mteb/NepaliNewsClassification")
    train_ds=dataset['train']
    test_ds=dataset['test']

    if train_samples:
        train_ds=train_ds.shuffle(seed=42).select(range(train_samples))
    if test_samples:
        test_ds=test_ds.shuffle(seed=42).select(range(test_samples))
    return train_ds,test_ds


def preprocess_text(dataset):
    return dataset["text"],dataset["label"]
