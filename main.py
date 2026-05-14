from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import math
import numpy as np
from groq import Groq
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REINDEX_PASSWORD = os.environ.get("REINDEX_PASSWORD", "comab@reindex2024")
DB_FILE = "./mips_db.json"

groq_client = Groq(api_key=GROQ_API_KEY)

# Banco simples em JSON
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": [], "embeddings": []}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)

# Embedding via Groq (modelo gratuito)
def get_embedding(text):
    # Usando llama para gerar representação semântica via prompt
    # Como Groq não tem endpoint de embedding, fazemos busca por similaridade textual simples
    return text.lower()

# Busca por similaridade textual simples (TF-IDF like)
def buscar_chunks(pergunta, chunks, n=4):
    palavras_pergunta = set(pergunta.lower().split())
    scores = []
    for i, chunk in enumerate(chunks):
        palavras_chunk = set(chunk["texto"].lower().split())
        intersecao = palavras_pergunta & palavras_chunk
        score = len(intersecao) / (math.sqrt(len(palavras_pergunta)) * math.sqrt(len(palavras_chunk)) + 1e-9)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:n] if scores[0][0] > 0]

# Quebrar texto em chunks
def chunk_text(texto, tamanho=400, overlap=50):
    palavras = texto.split()
    chunks = []
    i = 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i+tamanho])
        chunks.append(chunk)
        i += tamanho - overlap
    return chunks

class PerguntaRequest(BaseModel):
    pergunta: str

class ReindexRequest(BaseModel):
    senha: str

@app.post("/perguntar")
async def perguntar(req: PerguntaRequest):
    try:
        db = load_db()
        if not db["chunks"]:
            return {
                "resposta": "Ainda não tenho MIPs indexados. Por favor, faça o upload dos manuais primeiro.",
                "mip_consultado": None
            }

        chunks_relevantes = buscar_chunks(req.pergunta, db["chunks"])

        if not chunks_relevantes:
            return {
                "resposta": "Não encontrei informações sobre isso nos MIPs disponíveis.",
                "mip_consultado": None
            }

        contexto = "\n\n".join([c["texto"] for c in chunks_relevantes])
        mip_fonte = chunks_relevantes[0].get("arquivo", "MIP")

        prompt = f"""Você é o COMAB.IA-NO, assistente interno da COMAB Materiais de Construção.
Responda de forma clara, amigável e direta, baseando-se APENAS nas informações dos MIPs abaixo.
Se a resposta não estiver nos MIPs, diga que não encontrou essa informação nos manuais.
Nunca invente informações. Responda sempre em português brasileiro.

TRECHOS DOS MIPs:
{contexto}

PERGUNTA DO COLABORADOR:
{req.pergunta}

RESPOSTA:"""

        chat = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )

        resposta = chat.choices[0].message.content.strip()
        return {"resposta": resposta, "mip_consultado": mip_fonte}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindexar")
async def reindexar(req: ReindexRequest):
    if req.senha != REINDEX_PASSWORD:
        raise HTTPException(status_code=401, detail="Senha incorreta")

    try:
        mips_dir = "./mips"
        if not os.path.exists(mips_dir):
            os.makedirs(mips_dir)
            return {"status": "Pasta /mips criada. Adicione os arquivos .txt e reindexe novamente."}

        db = {"chunks": []}
        total = 0
        arquivos = []

        for filename in os.listdir(mips_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(mips_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    texto = f.read()

                chunks = chunk_text(texto)
                for chunk in chunks:
                    db["chunks"].append({
                        "texto": chunk,
                        "arquivo": filename.replace(".txt", "")
                    })

                total += len(chunks)
                arquivos.append(filename)

        save_db(db)
        return {
            "status": "Reindexação concluída!",
            "arquivos": arquivos,
            "total_chunks": total
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    db = load_db()
    return {"status": "ok", "chunks_indexados": len(db.get("chunks", []))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
