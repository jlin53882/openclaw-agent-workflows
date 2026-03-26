# OpenClaw Agent Workflows

> James 的 OpenClaw 多 Agent 協調與記憶系統實務記錄

---

## 檔案說明

### [COMMAND_CENTER_SETUP.md](COMMAND_CENTER_SETUP.md)
**幕僚長制度：多 Agent 協調中心架構**

如何建立一個「幕僚長」型態的協調 agent，透過 shared directory 與其他專門幕僚協作。

包含：
- 幕僚長角色定位
- shared/ 任務交接層設計
- openclaw.json 設定
- 各幕僚分工與派工 Protocol
- 常見問題

---

### [MEMORY_LANCEDB_PRO_SETUP.md](MEMORY_LANCEDB_PRO_SETUP.md)
**Memory LanceDB Pro 完整架構與參數調校指南**

如何架設本地部署的 RAG + 記憶系統，包含完整 pipeline、各環節參數與實測建議。

包含：
- 系統架構總覽（Ollama Embed → LanceDB → Rerank → Injection）
- 各環節延遲與瓶頸分析（實測數據）
- 模型選擇建議（Embedding + Reranker）
- 完整 reranker_server.py 範例
- 所有參數說明與建議值
- 調參決策指南
- 從零開始的安裝流程

---

## 授權

MIT License
