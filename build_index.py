import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from pypdf import PdfReader

#load environment variables from .env file
load_dotenv()
client = OpenAI()

chroma = PersistentClient(path="./rag_db")

collection = chroma.get_or_create_collection("mydocs")

#access API key
api_key = os.getenv("API_KEY")

def load_txt(path):
    return open(path, "r").read()

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text+=page.extract_text()+"\n"
    return text

def chunk_txt(txt, chunk_size=500):
    words = txt.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])
        
def load_all_documents(folder="Data"):
    documents = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        if filename.endswith(".pdf"):
            print("Loading PDF:", filename)
            documents.append(load_pdf(path))

        elif filename.endswith(".txt"):
            print("Loading TXT:", filename)
            documents.append(load_txt(path))

        else:
            print("Skipping unsupported file:", filename)
        
        # embed + store
    for doc in documents:
        for chunk in chunk_txt(doc):
            embedding = client.embeddings.create(
                model="text-embedding-3-large",
                input=chunk
            ).data[0].embedding

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(hash(chunk))]
            )

    return documents

load_all_documents()
print("RAG index built")