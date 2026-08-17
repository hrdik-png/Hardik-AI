from flask import Flask, render_template, request, jsonify, Response
from rag import retrieve, rewrite_query, groq_client

app = Flask(__name__)

conversation_history = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

    search_query = rewrite_query(
        question,
        conversation_history
    )

    retrieved_documents = retrieve(search_query)

    if not retrieved_documents:
        return jsonify({
            "answer": "I don't have enough information to answer that."
        })

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

    def generate():

        try:

            stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                stream=True
            )

            answer = ""

            for chunk in stream:

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    answer += content

                    yield content

            conversation_history.append({
                "role": "user",
                "content": question
            })

            conversation_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            yield f"\nAI request failed: {e}"

    return Response(
        generate(),
        mimetype="text/plain"
    )
@app.route("/clear", methods=["POST"])
def clear_chat():

    conversation_history.clear()

    return jsonify({
        "success": True
    })
if __name__ == "__main__":
    app.run(debug=False)