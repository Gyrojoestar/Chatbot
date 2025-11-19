import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

load_dotenv()
client = OpenAI()

chroma = PersistentClient(path="./rag_db")
collection = chroma.get_collection("mydocs")

print("Number of docs in collection:", collection.count())

def rag_answer(question):
    # 1) embed the question
    q_embed = client.embeddings.create(
        model="text-embedding-3-large",
        input=question
    ).data[0].embedding

    # 2) retrieve relevant chunks
    results = collection.query(
        query_embeddings=[q_embed],
        n_results=4
    )

    context = "\n\n".join(results["documents"][0])

    # 3) call GPT with context
    prompt = f"""
    You are a helpful AI assistant answering based on the context below.

    CONTEXT:
    {context}

    QUESTION: {question}

    If the answer is not in the context, say "I don't know."
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

if __name__ == "__main__":
    while True:
        q = input("You: ")
        print("Bot:", rag_answer(q))