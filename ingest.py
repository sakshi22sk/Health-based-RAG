import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_FILE = "vector.index"
DOC_FILE = "documents.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

# Read all txt files
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            documents.extend(chunks)

print(f"Total chunks: {len(documents)}")

# Create embeddings
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
np.save(DOC_FILE, documents)

print("✅ Ingestion completed successfully")