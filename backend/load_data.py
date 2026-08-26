import chromadb
import os
from dotenv import load_dotenv  # FIX: Import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pyTorch_data():
    load_dotenv()
    #this persistent client is saving to disk
    chroma_client = chromadb.PersistentClient(path="./chroma_db") #this is the database and the path is were it is being loaded, this saves onto the path so if something exists it will be loaded
    pytorch_collection = chroma_client.get_or_create_collection(
        name="pytorch-docs"
    ) 

    data_dir = "./data"
    

    documents = []
    metadatas = []
    ids = []

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

    
    for filename in os.listdir(data_dir):
        
        if filename.endswith('txt'):
            filepath = os.path.join(data_dir, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = splitter.split_text(f.read())
                for i in range(len(content)):
                    documents.append(content[i])
                    metadatas.append({"source": filename, "chunk": i})
                    ids.append(f"{filename.replace('.txt', '')}_chunk_{i}")
   

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

