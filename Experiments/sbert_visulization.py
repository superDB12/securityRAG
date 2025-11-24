import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# 1. Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Create a list of sentences with distinct topics
# We expect these to cluster into three groups: Fruits, Tech, and Weather.
sentences = [
    # Group 1: Fruits
    "The apple is juicy and red.",
    "I love eating fresh bananas.",
    "Oranges are rich in Vitamin C.",
    "Grapes make a great snack.",

    # Group 2: Technology
    "The new laptop has a fast processor.",
    "Artificial Intelligence is changing the world.",
    "My smartphone battery died.",
    "Python is a great programming language.",

    # Group 3: Weather
    "It is raining heavily outside.",
    "The sun is shining bright today.",
    "Snow is predicted for tomorrow.",
    "The wind is very strong this evening."
]

# 3. Generate Embeddings
print("Generating embeddings...")
embeddings = model.encode(sentences)

# 4. Reduce dimensions (384 -> 2) using t-SNE
# perplexity must be less than n_samples. For small data, we use a low value.
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
embeddings_2d = tsne.fit_transform(embeddings)

# 5. Plotting
plt.figure(figsize=(10, 8))

# Scatter plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', edgecolors='k', s=100)

# Add labels to points
for i, sentence in enumerate(sentences):
    plt.annotate(sentence,
                 (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 fontsize=9)

plt.title("t-SNE Visualization of Sentence Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()