# COMAB.IA-NO — Backend

Backend do assistente de MIPs da COMAB Materiais de Construção.

## Rotas

- `POST /perguntar` — recebe pergunta e retorna resposta baseada nos MIPs
- `POST /reindexar` — reindexar os MIPs da pasta /mips (requer senha)
- `GET /health` — health check para UptimeRobot

## Variáveis de ambiente

- `GROQ_API_KEY` — chave da API Groq
- `REINDEX_PASSWORD` — senha para reindexar os MIPs
