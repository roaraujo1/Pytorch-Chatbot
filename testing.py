import chromadb 
import os
from dotenv import load_dotenv  
from openai import OpenAI

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chromadb")