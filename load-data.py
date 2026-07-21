import chromadb 
import os
from dotenv import load_dotenv  # FIX: Import dotenv
from openai import OpenAI  # FIX: Import OpenAI client class
from fastapi import FastAPI

def load_pyTorch_data():
    load_dotenv()

    chroma_client = chromadb.PersistentClient(path="./chroma_db") #this is the database and the path is were it is being loaded, this saves onto the path so if something exists it will be loaded
    pytorch_collection = chroma_client.get_or_create_collection(
        name="pytorch-docs"
    ) 

    data_dir = "./data"

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(data_dir):
        print(filename)
        if filename.endswith('txt'):
            filepath = os.path.join(data_dir, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append(content)
            metadatas.append({"source": filename})
            ids.append(filename.replace('.txt', ''))

    if pytorch_collection.count() == 0:
        pytorch_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to collection")
    else:
        print(f"Collection already has {pytorch_collection.count()} documents, skipping load")

    print(f"Added {len(documents)} documents to collection")
    return pytorch_collection

pytorchData = load_pyTorch_data()

app = FastAPI()
@app.post("/ask")
async def ask(question: str):
    
    
   
    # FIX: Access the nested result correctly
    query_results = pytorchData.query(
        query_texts=[question],
        n_results=1
    )
    context = query_results['documents'][0][0]  # FIX: Added extra [0]

    print(f"Retrieved context (first 200 chars): {context[:200]}")

    # FIX: Create OpenAI client with correct name and syntax
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # FIX: Pass variable NAME not value

    prompt = f"{question}. Use this as context for answering: {context}"

    # FIX: Use the correct client
    response = openai_client.chat.completions.create(  # FIX: Changed from 'client' to 'openai_client'
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer: {answer}")
    return {"answer": answer}
