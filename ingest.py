import re

with open("user_info.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Major sections
sections = re.split(r"(?=---.*?---)", text)
sections = [section.strip() for section in sections if section.strip()]

chunks = []
# Technical subsections

technical_subsections = [
    "Programming Languages:",
    "Technical Skills:",
    "Tools/Technologies:",
    "Artificial Intelligence / Data Science:",
    "Projects:"
]
for section in sections:

    # Only split subsections inside Technical Background
    if section.startswith("---TECHNICAL BACKGROUND---"):

        subsection_pattern = (
            r"(?=" +
            "|".join(re.escape(name) for name in technical_subsections) +
            r")"
        )

        subsections = re.split(subsection_pattern, section)

        subsections = [
            subsection.strip()
            for subsection in subsections
            if subsection.strip()
        ]
       

        chunks.extend(subsections)
        

    else:
        chunks.append(section)

chunks = [
    chunk for chunk in chunks
    if chunk.strip() and not re.fullmatch(r"---.*?---", chunk.strip())
]
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Convert all chunks into embeddings
embeddings = model.encode(
    chunks,
    normalize_embeddings=True
).tolist()

# Open/create Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

# Start with a fresh collection while we're developing
try:
    client.delete_collection("hardik_info")
except:
    pass

collection = client.create_collection(
    name="hardik_info"
)

# Create one unique ID for every chunk
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Store everything
collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("Chunks stored in Chroma:", collection.count())
