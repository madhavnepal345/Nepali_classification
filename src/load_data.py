from datasets import load_dataset


def load_data(dataset_name,train_samples=None,test_samples=None):
    dataset=load_dataset(dataset_name)
    train_ds=dataset['train']
    test_ds=dataset['test']

    if train_samples:
        train_ds=train_ds.shuffle(seed=42).select(range(train_samples))
    if test_samples:
        test_ds=test_ds.shuffle(seed=42).select(range(test_samples))
    return train_ds,test_ds
