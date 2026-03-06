from data.load_data import load_dataset

docs, labels, cats = load_dataset()

print(len(docs))
print(cats)