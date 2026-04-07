# Memory LanceDB Pro 完整架構與參數調校指南

> 本文件說明如何架設、調校 memory-lancedb-pro（本地部署版），包含完整 pipeline、各環節參數意義與實測建議。
> 適用對象：有興趣架設本地 RAG + 記憶系統的 OpenClaw 部署。

---

## 1. 系統架構總覽

```
使用者輸入（query）
        ↓
┌─────────────────────────────────────────────┐
│         Ollama Embedding Service            │
│   http://localhost:11434/api/embeddings    │
│   Model: jina-v5-retrieval-test            │
│   用途：將 query 轉成向量                   │
└─────────────────────────────────────────────┘
        ↓ 向量
┌─────────────────────────────────────────────┐
│            LanceDB Hybrid Search            │
│   模式：vector + BM25 融合                 │
│   候選池：candidatePoolSize                 │
└─────────────────────────────────────────────┘
        ↓ 候補文件（通常 20-60 個）
┌─────────────────────────────────────────────┐
│         Rerank Server（本地）               │
│   http://localhost:18799/v1/rerank         │
│   Model: BAAI/bge-reranker-base            │
│   用途：重新排序，選出最相關的記憶           │
└─────────────────────────────────────────────┘
        ↓ 重排後結果
┌─────────────────────────────────────────────┐
│          autoRecall Injection               │
│   最終注入 agent context 的記憶數量         │
│   控制：autoRecallMaxItems                 │
└─────────────────────────────────────────────┘
```

---

## 2. 各環節延遲與瓶頸（實測數據）

**測試環境**：AMD Ryzen 7 8845HS (16 cores) / RAM 32GB DDR5 5600MT/s / 無 GPU / 記憶 4000+ 筆

| 環節 | 平均延遲 | 佔比 | 穩定性 |
|------|---------|------|--------|
| Ollama Embed（向量生成）| ~350ms | ~20% | CV=4.9%（非常穩）|
| LanceDB（hybrid search）| ~400ms | ~22% | 穩 |
| **Rerank Server** | **~2000ms** | **~55%** | **主要瓶頸** |
| **Pipeline 總計** | **~3600ms** | 100% | |

**2026-04-07 更新**：
- Rerank Server 實測 12 docs 約 2057ms（含 cache 寫入 34ms）
- 使用 `torch.set_num_threads(16)` 自動最大執行緒
- **批次處理已驗證有效** - 一次處理所有 doc，速度提升 6x（從 250ms/單doc 降到 40ms/單doc）

**關鍵發現**：
- Rerank 是最大瓶頸（~55% 延遲）
- 延遲與 doc 數量**幾乎無關**（批次處理優化後）
- Ollama embed 非常穩定，幾乎不需要優化
- CPU 在推理時使用率不高是正常（記憶體頻寬瓶頸，非 CPU 運算瓶頸）

---

## 3. 模型選擇建議

### 3.1 Embedding Model

| 模型 | 維度 | 大小 | 特色 | 推薦 |
|------|------|------|------|------|
| `jinaai/jina-embeddings-v5` | 1024 | ~400MB | 繁中最佳，量化版可用 | ✅ 推薦 |
| `jina-v5-retrieval-test` | 1024 | ~400MB | 測試版，效果相同 | ✅ 目前使用 |
| `nomic-embed-text` | 768 | ~270MB | 英文為主，繁中一般 | ⚠️ 不建議繁中 |
| `bge-m3` | 1024 | ~500MB | 多語言，但較慢 | ⚠️ CPU 負擔大 |

**推薦配置（Ollama）**：
```bash
# 安裝（使用量化版減少記憶體）
ollama pull jina-v5-retrieval-test

# 驗證
curl -X POST http://localhost:11434/api/embeddings \
  -d '{"model":"jina-v5-retrieval-test","prompt":"test"}'
```

### 3.2 Reranker Model

| 模型 | 大小 | 速度 | 準確度 | 推薦 |
|------|------|------|--------|------|
| `BAAI/bge-reranker-base` | ~400MB | 快（CPU 可跑）| 高 | ✅ 推薦本地部署 |
| `BAAI/bge-reranker-v2-m3` | ~1.1GB | 慢（5.8x）| 稍高 | ⚠️ 不建議，差異不大 |
| `jinaai/jina-reranker-v3` | ~1GB | 中 | 高 | ⚠️ 需要 API key |

**重要優化**：`BAAI/bge-reranker-base` 使用**批次處理**後，速度大幅提升（6x）。

---

## 4. Rerank Server 架設（Python 版）

### 4.1 安裝依賴

```bash
pip install transformers torch sentencepiece uvicorn fastapi pydantic httpx
```

### 4.2 啟動指令

```bash
# 基本啟動
python reranker_server.py

# 或者用 uvicorn 直接跑
uvicorn reranker_server:app --host 127.0.0.1 --port 18799
```

### 4.3 完整 reranker_server.py 範例（已優化）

```python
# -*- coding: utf-8 -*-
"""
本地 bge-reranker-base API Server
SiliconFlow 相容格式 /v1/rerank，供 memory-lancedb-pro 使用
"""
import os, time, torch, threading, asyncio, json, hashlib
from collections import OrderedDict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ============================================================
# 自動設定 CPU 執行緒數為最大值
# ============================================================
CPU_THREADS = os.cpu_count() or torch.get_num_threads()
torch.set_num_threads(CPU_THREADS)

# ============================================================
# 並發控制設定
# ============================================================
MAX_CONCURRENT = 100
BATCH_DELAY = 0.0          # 已停用

# ============================================================
# 快取設定（記憶體 + 磁碟）
# ============================================================
CACHE_MAX_ENTRIES = 500
cache_lock = threading.Lock()

def _make_cache_key(model_name, query, documents, top_n):
    raw = json.dumps({
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

RESULT_CACHE = OrderedDict()

# ============================================================
# 模型設定
# ============================================================
MODEL_NAME = "BAAI/bge-reranker-base"
PORT = 18799
DEVICE = torch.device("cpu")

print(f"正在載入模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print(f"模型載入完成，CPU 執行緒: {CPU_THREADS}")

# 預熱
dummy_inputs = tokenizer("warmup", "warmup", return_tensors="pt")
dummy_inputs = {k: v.to(DEVICE) for k, v in dummy_inputs.items()}
with torch.no_grad():
    model(**dummy_inputs)
print("預熱完成")

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI()
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: Optional[int] = None

@app.post("/v1/rerank")
async def rerank(request: Request):
    raw_body = await request.body()
    body_json = json.loads(raw_body)

    query = body_json.get("query", "")
    documents = body_json.get("documents")
    top_n = body_json.get("top_n")
    model_name = body_json.get("model", MODEL_NAME)

    if not documents or not query:
        return JSONResponse(status_code=400, content={"error": "query and documents required"})

    await semaphore.acquire()
    try:
        t_start = time.time()

        # 檢查 cache
        cache_key = _make_cache_key(model_name, query, documents, top_n)
        with cache_lock:
            cached = RESULT_CACHE.get(cache_key)
            if cached:
                RESULT_CACHE.move_to_end(cache_key)
                return cached

        # 批次處理（速度提升 6x）
        inputs = tokenizer(
            [query] * len(documents),
            documents,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        scores = torch.sigmoid(logits).squeeze(-1).cpu().tolist()

        # 確保是扁平列表
        if scores and isinstance(scores[0], list):
            scores = [s[0] for s in scores]

        # 排序
        indexed = [(i, documents[i], scores[i]) for i in range(len(documents))]
        indexed.sort(key=lambda x: x[2], reverse=True)
        if top_n:
            indexed = indexed[:top_n]

        results = [
            {"index": idx, "document": doc, "relevance_score": float(score)}
            for idx, doc, score in indexed
        ]

        payload = {"results": results}

        # 寫入 cache
        with cache_lock:
            RESULT_CACHE[cache_key] = payload
            RESULT_CACHE.move_to_end(cache_key)
            while len(RESULT_CACHE) > CACHE_MAX_ENTRIES:
                RESULT_CACHE.popitem(last=False)

        return payload
    finally:
        semaphore.release()

@app.get("/health")
async def health():
    return {"status": "ok", "cpu_threads": CPU_THREADS, "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
```

---

## 5. 完整參數說明

### 5.1 openclaw.json 設定結構

```json
{
  "plugins": {
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "autoCapture": true,
          "autoRecall": true,
          "dbPath": "~/.openclaw/memory/lancedb-pro-jina1024",

          "embedding": {
            "apiKey": "ollama-local",
            "baseURL": "http://localhost:11434/v1",
            "model": "jina-v5-retrieval-test",
            "dimensions": 1024,
            "normalized": true,
            "taskQuery": "retrieval.query",
            "taskPassage": "retrieval.passage"
          },

          "retrieval": {
            "mode": "hybrid",
            "vectorWeight": 0.6,
            "bm25Weight": 0.4,
            "candidatePoolSize": 30,
            "minScore": 0.25,
            "hardMinScore": 0.32,
            "rerank": "cross-encoder",
            "rerankProvider": "siliconflow",
            "rerankEndpoint": "http://127.0.0.1:18799/v1/rerank",
            "rerankApiKey": "local",
            "rerankModel": "BAAI/bge-reranker-base",
            "filterNoise": true,
            "lengthNormAnchor": 300,
            "recencyHalfLifeDays": 10,
            "recencyWeight": 0.06,
            "timeDecayHalfLifeDays": 45,
            "reinforcementFactor": 0.5,
            "maxHalfLifeMultiplier": 3
          },

          "autoRecall": true,
          "autoRecallTimeoutMs": 120000
        }
      }
    }
  }
}
```

### 5.2 參數對照表

#### Embedding 相關

| 參數 | 預設值 | 建議值 | 說明 |
|------|--------|--------|------|
| `embedding.model` | `jina-embeddings-v5` | `jina-v5-retrieval-test` | 繁中使用量化版省記憶體 |
| `embedding.dimensions` | 1024 | 1024 | 必須與模型維度一致 |
| `embedding.normalized` | true | true | 正規化向量，確保搜尋穩定 |

#### Retrieval 相關

| 參數 | 預設值 | 建議值 | 說明 |
|------|--------|--------|------|
| `mode` | `hybrid` | `hybrid` | 混用向量+BM25，最準 |
| `vectorWeight` | 0.7 | 0.6 | 向量搜尋權重 |
| `bm25Weight` | 0.3 | 0.4 | 關鍵字權重（繁中可高一點）|
| `candidatePoolSize` | 20 | 30 | 從 LanceDB 候選的數量 |
| `minScore` | 0.3 | 0.25 | BM25 最低分，設太高會漏結果 |
| `hardMinScore` | 0.35 | 0.32 | rerank 後的硬性過濾門檻 |

**`candidatePoolSize` 的真相**：這個數字控制從 LanceDB 取多少 candidate，不等於送給 rerank 的數量。實際送 rerank 的數量由 `autoRecallMaxItems * 2` 控制。

#### Rerank 相關

| 參數 | 預設值 | 建議值 | 說明 |
|------|--------|--------|------|
| `rerank` | `cross-encoder` | `cross-encoder` | 使用 reranker 重排序 |
| `rerankProvider` | `jina` | `siliconflow` | 本地部署用 siliconflow |
| `rerankEndpoint` | API URL | `http://127.0.0.1:18799/v1/rerank` | 本地 server |
| `rerankApiKey` | — | `local` | 本地不需要真的 key |
| `rerankModel` | `jina-reranker-v3` | `BAAI/bge-reranker-base` | 本地 CPU 模型 |

#### autoRecall 相關（plugin 內建）

| 參數 | 預設值 | 建議值 | 說明 |
|------|--------|--------|------|
| `autoRecall` | `true` | `true` | 啟用自動記憶注入 |
| `autoRecallMaxItems` | `3` | **維持 3** | 最終 injection 數量 |
| `autoRecallTimeoutMs` | `120000` | `120000` | 整個 recall 的逾時 |
| `retrieveLimit` | `6` | 自動（maxItems*2）| 實際送 rerank 的候選數 |

#### 時間衰減相關

| 參數 | 預設值 | 建議值 | 說明 |
|------|--------|--------|------|
| `recencyHalfLifeDays` | 14 | 10 | 新記憶加分（天數）|
| `recencyWeight` | 0.1 | 0.06 | 新記憶加分權重 |
| `timeDecayHalfLifeDays` | 60 | 45 | 舊記憶扣分（天數）|
| `reinforcementFactor` | 0.5 | 0.5 | 常用記憶延長半衰期 |
| `maxHalfLifeMultiplier` | 3 | 3 | 半衰期最多延長幾倍 |

---

## 6. 調參決策指南（實測後建議）

### 6.1 已經驗證不需要調整的參數

| 參數 | 原因 |
|------|------|
| `candidatePoolSize` | 對 rerank doc 數量無影響（由 retrieveLimit 控制）|
| `minScore` | 對此記憶庫無過濾效果（幾乎所有記憶 importance >= 0.5）|
| GPU 加速 | 成本太高，無 CUDA |
| Ollama embed 優化 | CV=4.9%，非常穩定 |

### 6.2 維持現狀的參數

| 參數 | 現值 | 理由 |
|------|------|------|
| `autoRecallMaxItems` | **3** | 2→3 升級有 84% 高相關記憶，邊際效益最佳 |
| `hardMinScore` | 0.32 | 再高會漏掉有效記憶 |
| `vectorWeight` | 0.6 | BM25 權重 0.4，適合繁中 |

### 6.3 已驗證有效的優化

| 優化 | 效果 | 狀態 |
|------|---------|------|
| `torch.set_num_threads(16)` | ~10-15% 提升 | ✅ 已驗證 |
| 批次處理 | 6x 提升 | ✅ 已驗證 |
| `torch.backends.mkldnn.enabled=True` | ~5-10% 提升 | 待驗證 |

### 6.4 為什麼批次處理有效

| 項目 | 說明 |
|------|------|
| 實現方式 | 一次 tokenize 所有 query-doc pairs，用 `model(**inputs)` 批次推理 |
| 速度提升 | 從 250ms/單doc 降到 40ms/單doc（12 docs 總計 ~2000ms） |
| 關鍵 | 使用 `torch.no_grad()` + `torch.sigmoid(logits).squeeze(-1)` |

---

## 7. Pipeline 延遲實測數據

### 單請求延遲（記憶庫 4000+ 筆）

| 場景 | 延遲 | 說明 |
|------|------|------|
| Ollama embed | ~350ms | 向量生成 |
| LanceDB search | ~400ms | hybrid search |
| Rerank 12 docs | ~2000ms | 含 cache 寫入 34ms |
| **Pipeline 總計** | **~3600ms** | |

### 時間分解（Rerank Server）

| 環節 | 時間 | 佔比 |
|------|------|------|
| cache_key (SHA256) | ~0.1ms | <1% |
| tokenize | ~6ms | <1% |
| to_device | ~1ms | <1% |
| **inference** | **~2000ms** | **~97%** |
| postproc + sort | ~0.5ms | <1% |
| cache_write | ~34ms | ~2% |

### 並發延遲

| 場景 | 延遲 |
|------|------|
| 並發 3 × 12 docs | ~2100ms |
| 並發 6 × 12 docs | ~2200ms |

---

## 8. 完整安裝流程（從零開始）

### Step 1：安裝 Ollama + Embedding 模型

```bash
# 安裝 Ollama（Windows/macOS）
# https://ollama.com/download

# 啟動 Ollama
ollama serve

# 安裝 embedding 模型
ollama pull jina-v5-retrieval-test

# 驗證
curl http://localhost:11434/api/tags
```

### Step 2：架設 Rerank Server

```bash
# 安裝 Python 依賴
pip install transformers torch sentencepiece uvicorn fastapi pydantic httpx

# 下載模型（會自動快取）
python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')"

# 啟動 server（會自動偵測 16 threads）
python reranker_server.py
```

### Step 3：設定 openclaw.json

```json
{
  "plugins": {
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "autoCapture": true,
          "autoRecall": true,
          "dbPath": "~/.openclaw/memory/lancedb-pro-local",
          "embedding": {
            "apiKey": "ollama-local",
            "baseURL": "http://localhost:11434/v1",
            "model": "jina-v5-retrieval-test",
            "dimensions": 1024,
            "normalized": true
          },
          "retrieval": {
            "mode": "hybrid",
            "vectorWeight": 0.6,
            "bm25Weight": 0.4,
            "candidatePoolSize": 30,
            "minScore": 0.25,
            "hardMinScore": 0.32,
            "rerank": "cross-encoder",
            "rerankProvider": "siliconflow",
            "rerankEndpoint": "http://127.0.0.1:18799/v1/rerank",
            "rerankApiKey": "local",
            "rerankModel": "BAAI/bge-reranker-base"
          }
        }
      }
    }
  }
}
```

### Step 4：重啟 Gateway

```bash
openclaw gateway restart
```

### Step 5：驗證

```bash
# 測試 embedding
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"jina-v5-retrieval-test","prompt":"test"}'

# 測試 rerank server
curl -X POST http://127.0.0.1:18799/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-reranker-base","query":"test","documents":["hello world","foo bar"]}'

# 觀察 server log 確認 cpu_threads=16
```

---

## 9. 常見問題

**Q：記憶 injection 有時多、有時少？**
A：正常現象。`hardMinScore=0.32` 會過濾掉低相關記憶，結果數量取決於查詢的相關文件數量。

**Q：auto-recall timeout 是什麼？**
A：整個 recall pipeline 的逾時限制（預設 120 秒）。通常不會觸發，但多 session 並發搶資源時可能發生。

**Q：需要 GPU 嗎？**
A：不需要。CPU 足以跑 `bge-reranker-base`（~400MB），延遲可接受（~2000ms for 12 docs）。

**Q：可以改用更強的 reranker 嗎？**
A：可以但收益有限。`bge-reranker-v2-m3` 準確度稍高，但速度慢 5.8x，且需要更大記憶體，性價比不佳。

**Q：CPU 使用率不高是正常的嗎？**
A：正常。Transformer 推理是記憶體頻寬瓶頸（memory-bound），不是 CPU 運算瓶頸。大部分時間在等待記憶體載入權重。

**Q：batch_size 要設多少？**
A：不需要設定。Server 會一次處理所有 documents，自動最佳化。

---

*本文件基於 2026-04-07 更新測試結果編寫*
*更新內容：批次處理驗證、torch.set_num_threads(16) 驗證、延遲數據更新*
