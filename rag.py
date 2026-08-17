import os
import re
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# Groq
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Embedding model
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# Chroma
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="hardik_info"
)
def retrieve(query):

    instruction = "Represent this sentence for searching relevant passages: "

    query_embedding = embedding_model.encode(
        instruction + query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=12
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "does", "do", "did", "has", "have",
        "how", "who", "where", "when", "why",
        "which", "and", "or", "of", "to",
        "in", "on", "for", "with", "from"
    }

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

        final_score = semantic_score + 0.25*keyword_score

        ranked_results.append(
            (document, distance, final_score)
        )

    ranked_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    retrieved_documents = [
        document
        for document, distance, score in ranked_results[:3]
    ]
    print("\n--- RETRIEVAL ---")

    for document, distance, score in ranked_results[:3]:
        print(f"Score: {score:.4f} | Distance: {distance:.4f}")
        print(document[:150])
        print()
    return retrieved_documents

def rewrite_query(question, conversation_history):

    search_query = question

    if conversation_history:
        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_history
        )

        rewrite_prompt = f"""
        Rewrite the user's latest question into a standalone search query.

        Use the conversation history only to resolve references such as:
        - it
        - they
        - them
        - this
        - that
        - which one

        Do not answer the question.
        Do not add information that isn't present in the conversation.

        Conversation history:
        {history_text}

        Latest question:
        {question}

        Standalone search query:
        """

        try:
            rewrite_response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": rewrite_prompt
                    }
                ]
            )

            search_query = rewrite_response.choices[0].message.content.strip()

        except Exception as e:
            print(f"\nQuery rewriting failed: {e}")
            search_query = question

    return search_query

def main():
    conversation_history = []

    while True:
        question = input("You: ")
        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        search_query = rewrite_query(
            question,
            conversation_history
        )

        retrieved_documents = retrieve(search_query)
    
        if not retrieved_documents:
            print("\nAI: I don't have enough information to answer that.")
            continue
        context = "\n\n".join(retrieved_documents)
        prompt = f"""
        You are answering questions about Hardik.

        Use ONLY the information provided in the context below.

        Rules:
        - Answer the user's question using only the context.
        - Treat the context as the only source of factual information.
        - Do not make up information.
        - Do not infer or assume facts that are not explicitly stated in the context.
        - If the context does not explicitly contain enough information to answer the question, say:
        "I don't have enough information to answer that."
        - Keep the answer concise and natural.

        Context:
        {context}

        Question:
        {question}
        """
    

        messages = conversation_history + [
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            stream = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True
            )

            answer = ""

            print("\nAI:", end=" ")

            for chunk in stream:

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    print(content, end="", flush=True)
                    answer += content

            print()

        except Exception as e:
            print(f"\nAI request failed: {e}")
            continue


        conversation_history.append({
            "role": "user",
            "content": question
        })
        conversation_history.append({
        "role": "assistant",
        "content": answer
        })
    
if __name__ == "__main__":
    main()
