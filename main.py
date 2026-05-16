from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import math
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
MIPS_DIR = "./mips"

groq_client = Groq(api_key=GROQ_API_KEY)

# Banco em memória — carregado na inicialização
chunks_memoria = []

def chunk_text(texto, tamanho=400, overlap=50):
    palavras = texto.split()
    chunks = []
    i = 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i+tamanho])
        chunks.append(chunk)
        i += tamanho - overlap
    return chunks

def carregar_mips():
    global chunks_memoria
    chunks_memoria = []

    if not os.path.exists(MIPS_DIR):
        print("Pasta /mips não encontrada.")
        return

    arquivos = [f for f in os.listdir(MIPS_DIR) if f.endswith(".txt")]
    if not arquivos:
        print("Nenhum MIP encontrado na pasta /mips.")
        return

    for filename in arquivos:
        filepath = os.path.join(MIPS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            texto = f.read()
        chunks = chunk_text(texto)
        nome = filename.replace(".txt", "").replace("_", " ")
        for chunk in chunks:
            chunks_memoria.append({"texto": chunk, "arquivo": nome})

    print(f"✅ {len(chunks_memoria)} chunks carregados de {len(arquivos)} MIP(s): {', '.join(arquivos)}")

def buscar_chunks(pergunta, n=5):
    if not chunks_memoria:
        return []
    palavras_pergunta = set(pergunta.lower().split())
    scores = []
    for i, chunk in enumerate(chunks_memoria):
        palavras_chunk = set(chunk["texto"].lower().split())
        intersecao = palavras_pergunta & palavras_chunk
        score = len(intersecao) / (math.sqrt(len(palavras_pergunta)) * math.sqrt(len(palavras_chunk)) + 1e-9)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [chunks_memoria[i] for score, i in scores[:n] if score > 0]

# Carregar MIPs ao iniciar
@app.on_event("startup")
async def startup():
    carregar_mips()

class PerguntaRequest(BaseModel):
    pergunta: str

@app.post("/perguntar")
async def perguntar(req: PerguntaRequest):
    try:
        if not chunks_memoria:
            return {
                "resposta": "Os Manuais Internos ainda não foram carregados. Por favor, aguarde ou contate o administrador.",
                "mip_consultado": None
            }

        chunks_relevantes = buscar_chunks(req.pergunta)

        if not chunks_relevantes:
            return {
                "resposta": "Não encontrei informações sobre isso nos MIPs disponíveis. Tente reformular a pergunta ou consulte seu gestor.",
                "mip_consultado": None
            }

        contexto = "\n\n---\n\n".join([c["texto"] for c in chunks_relevantes])

        # MIPs consultados (sem repetição)
        mips_usados = list(dict.fromkeys([c["arquivo"] for c in chunks_relevantes]))
        mip_fonte = ", ".join(mips_usados)

        prompt = f"""Você é o COMAB.IA-NO, assistente interno da COMAB Materiais de Construção.

PERSONALIDADE:
- Você é como um colega de trabalho experiente, engraçado na medida certa e nunca robótico
- Chame SEMPRE o usuário de "Comabiano" — isso cria senso de pertencimento
- Use linguagem natural e brasileira: "bora", "tranquilo", "show", "certinho"
- Faça piadas leves e pertinentes ao contexto — especialmente sobre o dia a dia do varejo
- Celebre quando resolver algo complexo: "Missão cumprida! 💪"
- Incentive: "Qualquer dúvida é só chamar!"
- Use emojis com moderação — 1 a 2 por resposta, só quando reforçam o tom

EXEMPLOS DE TOM:
- Processo longo: "Esse processo tem mais etapas do que segunda-feira tem problema, mas bora lá, Comabiano! 😄"
- Não encontrou: "Ei Comabiano, queria muito te ajudar nisso, mas isso não tá no meu território não! 😅 Melhor confirmar com seu gestor."
- Regra importante: "Nunca, Comabiano! ❌ Isso é passível de desligamento — sua senha é sua, como escova de dente, não se empresta! 😬"
- Pergunta vaga: "Hmm, Comabiano... 'aquele negócio lá' tem muitas possibilidades! Me conta mais — é sobre pedido, entrega, precificação?"
- Prazo/alerta: use ⏰ ou 🚨 para reforçar urgência
- Após resposta longa: sempre termine com "Se tiver dúvida em algum passo, é só perguntar, Comabiano!"
- Elogio recebido: "Fico feliz em ajudar! 😄 Tô aqui plantado nos MIPs esperando suas perguntas!"

REGRAS INVIOLÁVEIS:
- Responda APENAS com base nas informações dos MIPs abaixo — nunca invente
- Se não encontrar a resposta, admita com bom humor e sugira consultar o gestor
- Se o processo tiver passos, liste-os organizados com negrito nos pontos importantes
- Responda sempre em português brasileiro
- Nunca seja rude, grosseiro ou desmotivador

TRECHOS DOS MIPs:
{contexto}

PERGUNTA DO COMABIANO:
{req.pergunta}

RESPOSTA DO COMAB.IA-NO:"""

        chat = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2
        )

        resposta = chat.choices[0].message.content.strip()
        return {"resposta": resposta, "mip_consultado": mip_fonte}

    except Exception as e:
        import traceback
        print(f"ERRO /perguntar: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    arquivos = list(dict.fromkeys([c["arquivo"] for c in chunks_memoria]))
    return {
        "status": "ok",
        "chunks_carregados": len(chunks_memoria),
        "mips": arquivos
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
