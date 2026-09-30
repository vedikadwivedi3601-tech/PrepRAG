from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Precision measures how many predicted positives are actually positive.",
    "Recall measures how many actual positives were correctly identified.",
    "A decision tree is a supervised machine learning algorithm."
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)
print("\nFirst sentence embedding:")
print(embeddings[0])