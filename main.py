from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import uvicorn

app = FastAPI()

# CORS — permite o frontend acessar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REINDEX_PASSWORD = os.environ.get("REINDEX_PASSWORD", "comab@reindex2024")

# ChromaDB — banco vetorial local
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = chroma_client.get_or_create_collection(
    name="mips",
    embedding_function=embedding_fn
)

groq_client = Groq(api_key=GROQ_API_KEY)

# Models
class PerguntaRequest(BaseModel):
    pergunta: str

class ReindexRequest(BaseModel):
    senha: str

# Função para quebrar texto em chunks
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# Rota principal — responder perguntas
@app.post("/perguntar")
async def perguntar(req: PerguntaRequest):
    try:
        # Buscar trechos relevantes no ChromaDB
        results = collection.query(
            query_texts=[req.pergunta],
            n_results=4
        )

        if not results["documents"][0]:
            return {
                "resposta": "Ainda não tenho MIPs indexados. Por favor, faça o upload dos manuais primeiro.",
                "mip_consultado": None
            }

        # Montar contexto com os trechos encontrados
        contexto = "\n\n".join(results["documents"][0])
        metadatas = results["metadatas"][0]
        mip_fonte = metadatas[0].get("arquivo", "MIP") if metadatas else "MIP"

        # Prompt para a IA
        prompt = f"""Você é o COMAB.IA-NO, assistente interno da COMAB Materiais de Construção.
Responda de forma clara, amigável e direta, baseando-se APENAS nas informações dos MIPs abaixo.
Se a resposta não estiver nos MIPs, diga que não encontrou essa informação nos manuais.
Nunca invente informações. Responda sempre em português brasileiro.

TRECHOS DOS MIPs:
{contexto}

PERGUNTA DO COLABORADOR:
{req.pergunta}

RESPOSTA:"""

        # Chamar a Groq API
        chat = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )

        resposta = chat.choices[0].message.content.strip()

        return {
            "resposta": resposta,
            "mip_consultado": mip_fonte
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rota para upload e indexação de MIPs
@app.post("/reindexar")
async def reindexar(req: ReindexRequest):
    if req.senha != REINDEX_PASSWORD:
        raise HTTPException(status_code=401, detail="Senha incorreta")

    try:
        mips_dir = "./mips"
        if not os.path.exists(mips_dir):
            return {"status": "Nenhum MIP encontrado na pasta /mips"}

        # Limpar coleção atual
        chroma_client.delete_collection("mips")
        global collection
        collection = chroma_client.get_or_create_collection(
            name="mips",
            embedding_function=embedding_fn
        )

        total_chunks = 0
        arquivos_processados = []

        for filename in os.listdir(mips_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(mips_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    texto = f.read()

                chunks = chunk_text(texto)

                ids = [f"{filename}_{i}" for i in range(len(chunks))]
                metadatas = [{"arquivo": filename.replace(".txt", "")} for _ in chunks]

                collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )

                total_chunks += len(chunks)
                arquivos_processados.append(filename)

        return {
            "status": "Reindexação concluída!",
            "arquivos": arquivos_processados,
            "total_chunks": total_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check — para o UptimeRobot
@app.get("/health")
async def health():
    return {"status": "ok", "mips_indexados": collection.count()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
