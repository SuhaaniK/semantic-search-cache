from embeddings.embedder import Embedder

model = Embedder()

vec = model.embed(["hello world"])

print(vec.shape)