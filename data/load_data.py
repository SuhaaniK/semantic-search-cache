from sklearn.datasets import fetch_20newsgroups
from utils.preprocess import clean_text

def load_dataset():

    dataset = fetch_20newsgroups(subset="all")

    documents = [clean_text(d) for d in dataset.data]
    labels = dataset.target
    categories = dataset.target_names

    return documents, labels, categories
    # documents = dataset.data
    # labels = dataset.target
    # categories = dataset.target_names

    # return documents, labels, categories