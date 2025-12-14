from datasets import load_dataset


def load_data(train_samples=200000,test_samples=50000):
    dataset=load_dataset("IRIIS-RESEARCH/Nepali-Text-Corpus")
    train_ds=dataset['train']
    test_ds=dataset['test']

    if train_samples:
        train_ds=train_ds.shuffle(seed=42).select(range(train_samples))
    if test_samples:
        test_ds=test_ds.shuffle(seed=42).select(range(test_samples))
    return train_ds,test_ds


def preprocess_text(dataset):
    return dataset["Article"],dataset["Source"]
