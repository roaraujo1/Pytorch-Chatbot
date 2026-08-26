import os
from load_data import load_pyTorch_data
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI  # FIX: Import OpenAI client class
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
pytorchData = load_pyTorch_data()

app = FastAPI()
origins = [
    "http://localhost:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods =["*"],
    allow_headers=["*"],

)
class QuestionRequest(BaseModel):
    question:str

@app.post("/ask")
async def ask(req: QuestionRequest):
    
    query_results = pytorchData.query(
        query_texts=[req.question],
        n_results=1
    )
    context = query_results['documents'][0][0]  # FIX: Added extra [0]
    

    # FIX: Create OpenAI client with correct name and syntax
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # FIX: Pass variable NAME not value

    prompt = f"{req.question}. Use this as context for answering: {context}"

    # FIX: Use the correct client
    response = openai_client.chat.completions.create(  # FIX: Changed from 'client' to 'openai_client'
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
   
    return {"answer": answer}
