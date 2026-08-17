import chromadb
import re
stop_words = {
    "what", "is", "are", "the", "a", "an",
    "does", "do", "did", "has", "have",
    "how", "who", "where", "when", "why",
    "which", "and", "or", "of", "to",
    "in", "on", "for", "with", "from"
}
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="hardik_info"
)

query = input("Question: ")
query = query.replace("Hardik", "").strip()
query = " ".join(query.split())
instruction = "Represent this sentence for searching relevant passages: "

query_embedding = model.encode(
    instruction + query,
    normalize_embeddings=True
).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=12
)

documents = results["documents"][0]
distances = results["distances"][0]

# Words from the user's query
query_words = {
    word
    for word in re.findall(r"\b\w+\b", query.lower())
    if word not in stop_words
}
ranked_results = []



for document, distance in zip(documents, distances):

    document_words = set(
        re.findall(r"\b\w+\b", document.lower())
    )

    keyword_matches = query_words & document_words

    keyword_score = len(keyword_matches)

    semantic_score = 1 / (1 + distance)

    final_score = semantic_score + keyword_score

    ranked_results.append(
        (document, distance, keyword_score, final_score)
    )

ranked_results.sort(
    key=lambda x: x[3],
    reverse=True
)

print("\nHybrid retrieval results:")

for document, distance, keyword_score, final_score in ranked_results[:3]:

    print(f"\nDistance: {distance:.4f}")
    print(f"Keyword matches: {keyword_score}")
    print(f"Final score: {final_score:.4f}")
    print(document)