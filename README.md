# PromptShield MCP

Adversarial prompt detection & self-repair — FastAPI backend + React dashboard.

---

## Proje Yapısı

```
promptshield_mcp/          ← backend (Python)
├── api.py
├── agentic_loop.py
├── config.py
├── requirements.txt
├── agents/
├── models/
├── utils/
├── mcp_tools/
├── dataset/data/
└── evaluation/

dashboard.jsx              ← React frontend (tek dosya)
render.yaml                ← Render deploy config
README.md
```

---

## 1 — Yerel Çalıştırma

### Backend
```bash
cd promptshield_mcp
pip install -r requirements.txt
uvicorn api:app --reload
```
API: http://localhost:8000  
Docs: http://localhost:8000/docs

### Dashboard
```bash
npm create vite@latest promptshield-ui -- --template react
cd promptshield-ui
npm install
cp ../dashboard.jsx src/App.jsx
npm run dev
```
→ http://localhost:5173

---

## 2 — Render.com Ücretsiz Deploy

**1. GitHub'a yükle**
```bash
git init && git add . && git commit -m "initial"
git remote add origin https://github.com/KULLANICI/promptshield-mcp.git
git push -u origin main
```

**2. render.com → New → Web Service → GitHub repoyu bağla**

| Alan | Değer |
|------|-------|
| Root Directory | `promptshield_mcp` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Plan | Free |

**3. Deploy bittikten sonra dashboard.jsx içinde URL güncelle:**
```js
const API = "https://promptshield-mcp-xxxx.onrender.com";
```

---

## API

| Method | URL | Ne yapar |
|--------|-----|----------|
| POST | `/analyze` | Sadece Detection Agent (hızlı) |
| POST | `/run` | Tam 4-ajan pipeline |
| GET | `/dataset?split=test` | Dataset kayıtları |
| GET | `/dataset/stats` | İstatistikler |
| GET | `/evaluation/summary` | Sistem karşılaştırması |
| GET | `/evaluation/categories` | Kategori bazlı sonuçlar |

