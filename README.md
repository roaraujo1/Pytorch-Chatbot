# PyTorch Docs Chatbot

A retrieval-augmented Q&A chatbot that answers questions about PyTorch using its official documentation. Built with a FastAPI + ChromaDB backend and a React (Vite) frontend.

Ask a question in plain English (e.g. *"How do I create a custom Dataset class in PyTorch?"*) and the app retrieves the most relevant passage from the PyTorch docs, feeds it to an LLM as context, and returns a grounded, conversational answer — instead of relying purely on the model's training data.

## How it works

1. **Ingestion** — `.txt` files scraped from the official PyTorch documentation are loaded from `data/` and embedded into a persistent ChromaDB vector store (`chroma_db/`) on server startup.
2. **Retrieval** — when a question comes in, the backend queries ChromaDB for the most relevant document chunk.
3. **Generation** — that chunk is passed as context to OpenAI's `gpt-4o` alongside the user's question, and the model's answer is returned to the frontend.
4. **Display** — the React frontend renders the response as formatted Markdown (headers, code blocks, bold text) instead of raw text.

This is a standard RAG (Retrieval-Augmented Generation) pipeline — the same pattern used in production documentation assistants and internal knowledge-base tools.

## Tech stack

**Backend:** FastAPI, ChromaDB, OpenAI API, Pydantic, Uvicorn

**Frontend:** React, Vite, Axios, react-markdown

## Project structure

```
Pytorch-Chatbot/
├── backend/
│   ├── load_data.py       # FastAPI app: ingestion + /ask endpoint
│   ├── chroma_db/         # persisted vector store (gitignored)
│   ├── data/               # source .txt files from PyTorch docs
│   ├── requirements.txt
│   └── .env                # OPENAI_API_KEY (gitignored)
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js           # axios instance pointing at the backend
    │   └── components/
    │       └── Question.jsx # question input + answer display
    └── package.json
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI API key

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/` (see `.env.example`):

```
OPENAI_API_KEY=your_openai_api_key_here
```

Run the server:

```bash
uvicorn load_data:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

> Note: the backend's CORS settings must allow the frontend's origin (`http://localhost:5173` by default) — see `origins` in `load_data.py`.

## API

### `POST /ask`

**Request body:**

```json
{ "question": "What is autograd in PyTorch?" }
```

**Response:**

```json
{ "answer": "Autograd is PyTorch's automatic differentiation engine..." }
```

## Roadmap / possible next steps

- Conversation history (multi-turn chat instead of single Q&A)
- Deployment (backend on Render/Railway, frontend on Vercel)
- Automated re-ingestion when new docs are added to `data/`
- Basic tests for the `/ask` endpoint

