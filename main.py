from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
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
REINDEX_PASSWORD = os.environ.get("REINDEX_PASSWORD", "comab@reindex2024")
DB_FILE = "./mips_db.json"
MIPS_DIR = "./mips"

groq_client = Groq(api_key=GROQ_API_KEY)

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": []}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)

def chunk_text(texto, tamanho=400, overlap=50):
    palavras = texto.split()
    chunks = []
    i = 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i+tamanho])
        chunks.append(chunk)
        i += tamanho - overlap
    return chunks

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

class PerguntaRequest(BaseModel):
    pergunta: str

class ReindexRequest(BaseModel):
    senha: str

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    db = load_db()
    total = len(db.get("chunks", []))
    arquivos = list(set([c["arquivo"] for c in db.get("chunks", [])]))
    arquivos_html = "".join([f"<li>📄 {a}</li>" for a in arquivos]) or "<li>Nenhum MIP indexado ainda</li>"

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>COMAB.IA-NO — Admin</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; background: #080F1E; color: white; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }}
  .card {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 24px; padding: 40px; width: 100%; max-width: 560px; backdrop-filter: blur(20px); }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  h1 span {{ color: #FF7070; }}
  .sub {{ color: rgba(255,255,255,0.4); font-size: 13px; margin-bottom: 32px; }}
  .section {{ margin-bottom: 28px; }}
  .section h2 {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: rgba(255,255,255,0.4); margin-bottom: 12px; }}
  .stats {{ display: flex; gap: 12px; margin-bottom: 24px; }}
  .stat {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; flex: 1; text-align: center; }}
  .stat .val {{ font-size: 28px; font-weight: 700; color: #2979FF; }}
  .stat .lbl {{ font-size: 11px; color: rgba(255,255,255,0.4); margin-top: 2px; }}
  ul {{ list-style: none; background: rgba(255,255,255,0.04); border-radius: 12px; padding: 12px 16px; }}
  ul li {{ font-size: 13px; color: rgba(255,255,255,0.7); padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
  ul li:last-child {{ border-bottom: none; }}
  .drop-area {{ border: 2px dashed rgba(255,255,255,0.2); border-radius: 14px; padding: 32px; text-align: center; cursor: pointer; transition: all 0.2s; }}
  .drop-area:hover, .drop-area.drag {{ border-color: #2979FF; background: rgba(41,121,255,0.08); }}
  .drop-area p {{ color: rgba(255,255,255,0.5); font-size: 13px; margin-top: 8px; }}
  .drop-icon {{ font-size: 32px; }}
  input[type=file] {{ display: none; }}
  input[type=password] {{ width: 100%; padding: 12px 16px; background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; color: white; font-family: 'Inter', sans-serif; font-size: 14px; outline: none; margin-bottom: 12px; }}
  input[type=password]::placeholder {{ color: rgba(255,255,255,0.3); }}
  button {{ width: 100%; padding: 13px; background: linear-gradient(135deg, #1B3A7A, #C0202A); border: none; border-radius: 12px; color: white; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }}
  button:hover {{ opacity: 0.9; }}
  button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  .msg {{ margin-top: 12px; padding: 12px 16px; border-radius: 10px; font-size: 13px; display: none; }}
  .msg.ok {{ background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.3); color: #86efac; }}
  .msg.err {{ background: rgba(192,32,42,0.15); border: 1px solid rgba(192,32,42,0.3); color: #ffaaaa; }}
  .file-list {{ margin-top: 10px; font-size: 12px; color: rgba(255,255,255,0.5); }}
</style>
</head>
<body>
<div class="card">
  <h1>COMAB<span>.IA-NO</span></h1>
  <p class="sub">Painel de Administração — Upload de MIPs</p>

  <div class="stats">
    <div class="stat"><div class="val">{total}</div><div class="lbl">Chunks indexados</div></div>
    <div class="stat"><div class="val">{len(arquivos)}</div><div class="lbl">MIPs carregados</div></div>
  </div>

  <div class="section">
    <h2>MIPs Indexados</h2>
    <ul>{arquivos_html}</ul>
  </div>

  <div class="section">
    <h2>Upload de Novo MIP</h2>
    <div class="drop-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
      <div class="drop-icon">📄</div>
      <strong>Clique ou arraste arquivos .txt aqui</strong>
      <p>Apenas arquivos .txt estruturados</p>
    </div>
    <input type="file" id="fileInput" accept=".txt" multiple>
    <div class="file-list" id="fileList"></div>
  </div>

  <div class="section">
    <h2>Senha de Administrador</h2>
    <input type="password" id="senha" placeholder="Digite a senha de admin" />
    <button id="uploadBtn" onclick="uploadMIPs()">⬆️ Fazer Upload e Reindexar</button>
    <div class="msg" id="msg"></div>
  </div>
</div>

<script>
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');
  const fileList = document.getElementById('fileList');
  let selectedFiles = [];

  dropArea.addEventListener('dragover', e => {{ e.preventDefault(); dropArea.classList.add('drag'); }});
  dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag'));
  dropArea.addEventListener('drop', e => {{
    e.preventDefault(); dropArea.classList.remove('drag');
    selectedFiles = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.txt'));
    updateFileList();
  }});
  fileInput.addEventListener('change', () => {{
    selectedFiles = Array.from(fileInput.files);
    updateFileList();
  }});

  function updateFileList() {{
    fileList.textContent = selectedFiles.length > 0
      ? '📎 ' + selectedFiles.map(f => f.name).join(', ')
      : '';
  }}

  async function uploadMIPs() {{
    const senha = document.getElementById('senha').value;
    const msg = document.getElementById('msg');
    const btn = document.getElementById('uploadBtn');

    if (!senha) {{ showMsg('Digite a senha de administrador.', false); return; }}
    if (selectedFiles.length === 0) {{ showMsg('Selecione ao menos um arquivo .txt.', false); return; }}

    btn.disabled = true;
    btn.textContent = '⏳ Enviando e indexando...';

    try {{
      const formData = new FormData();
      formData.append('senha', senha);
      selectedFiles.forEach(f => formData.append('arquivos', f));

      const res = await fetch('/upload-mips', {{ method: 'POST', body: formData }});
      const data = await res.json();

      if (res.ok) {{
        showMsg('✅ ' + data.status + ' — ' + data.total_chunks + ' chunks indexados de ' + data.arquivos.length + ' arquivo(s).', true);
        setTimeout(() => location.reload(), 2000);
      }} else {{
        showMsg('❌ ' + (data.detail || 'Erro desconhecido'), false);
      }}
    }} catch(e) {{
      showMsg('❌ Erro de conexão: ' + e.message, false);
    }}

    btn.disabled = false;
    btn.textContent = '⬆️ Fazer Upload e Reindexar';
  }}

  function showMsg(text, ok) {{
    const msg = document.getElementById('msg');
    msg.textContent = text;
    msg.className = 'msg ' + (ok ? 'ok' : 'err');
    msg.style.display = 'block';
  }}
</script>
</body>
</html>"""

@app.post("/upload-mips")
async def upload_mips(senha: str = None, arquivos: list[UploadFile] = File(...)):
    if senha != REINDEX_PASSWORD:
        raise HTTPException(status_code=401, detail="Senha incorreta")

    os.makedirs(MIPS_DIR, exist_ok=True)

    # Salvar arquivos enviados
    for arquivo in arquivos:
        if not arquivo.filename.endswith(".txt"):
            continue
        conteudo = await arquivo.read()
        caminho = os.path.join(MIPS_DIR, arquivo.filename)
        with open(caminho, "wb") as f:
            f.write(conteudo)

    # Reindexar tudo
    db = {"chunks": []}
    total = 0
    nomes = []

    for filename in os.listdir(MIPS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(MIPS_DIR, filename), "r", encoding="utf-8") as f:
                texto = f.read()
            chunks = chunk_text(texto)
            for chunk in chunks:
                db["chunks"].append({"texto": chunk, "arquivo": filename.replace(".txt", "")})
            total += len(chunks)
            nomes.append(filename)

    save_db(db)
    return {"status": "Upload e reindexação concluídos!", "arquivos": nomes, "total_chunks": total}

@app.post("/perguntar")
async def perguntar(req: PerguntaRequest):
    try:
        db = load_db()
        if not db["chunks"]:
            return {"resposta": "Ainda não tenho MIPs indexados. Por favor, faça o upload dos manuais primeiro.", "mip_consultado": None}

        chunks_relevantes = buscar_chunks(req.pergunta, db["chunks"])
        if not chunks_relevantes:
            return {"resposta": "Não encontrei informações sobre isso nos MIPs disponíveis.", "mip_consultado": None}

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

        return {"resposta": chat.choices[0].message.content.strip(), "mip_consultado": mip_fonte}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    db = load_db()
    return {"status": "ok", "chunks_indexados": len(db.get("chunks", []))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
