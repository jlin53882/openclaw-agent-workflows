# 幕僚長制度：OpenClaw 多 Agent 協調中心架構

> 本文件說明如何建立一個「幕僚長」型態的協調 agent，透過 shared directory 與其他專門幕僚協作。
> 適用對象：有多個獨立 agent、需要協調分工的 OpenClaw 部署。

---

## 概念說明

### 問題
- 多個 agent 各自獨立運作，缺少統一的協調層
- 沒有任務追蹤，難以確認「誰在做什麼」
- 任務分配靠口頭或直接對話，沒有結構

### 解法：幕僚長制度
```
James（指揮）
     ↓
幕僚長（協調者）
     ↓
幕僚 A（實作）｜幕僚 B（測試）｜幕僚 C（文修）
     ↓
結果寫入 shared/results/
```

幕僚長不自己做實作，專門負責：
1. 接收指令 → 拆解成任務
2. 派工給對應幕僚
3. 追蹤進度
4. 彙整結果回報

---

## 目錄架構

```
~/.openclaw/workspace/
├── shared/                           ← 所有 agent 共享（必備）
│   ├── tasks/                        ← 任務佇列（指揮中心寫，幕僚讀）
│   │   └── <taskId>.json
│   ├── results/                      ← 結果交付（幕僚寫，指揮中心讀）
│   │   └── <taskId>__<agentId>-result.json
│   ├── status/                       ← 狀態追蹤
│   │   └── <taskId>.status.json
│   └── dispatch_task.py              ← 派工 CLI 工具（可選）
│
├── workspace-<agent-name>/            ← 各幕僚的獨立 workspace
│   ├── SOUL.md                       ← 角色定義
│   ├── AGENTS.md                     ← 協調 protocol
│   ├── MEMORY.md                     ← 專屬記憶
│   ├── memory/
│   └── skills/
│
└── docs/
    └── COMMAND_CENTER_SETUP.md       ← 本文件
```

---

## Step 1：建立幕僚長的 workspace

```bash
mkdir workspace-<agent-id>
```

建立 `SOUL.md`，定義幕僚長角色：

```markdown
# SOUL.md — 幕僚長

> 你是 James 的幕僚長，負責協調整個幕僚團隊的運作。

## 核心原則
- 不自己下場：複雜問題找對應的專家 agent
- 問對問題：派任務時確認成功標準
- 追蹤到底：任務發出去後主動追蹤直到有結果

## 任務交接流程
詳細見 `AGENTS.md`。快速摘要：
1. 建立 shared/tasks/<taskId>.json
2. 建立 shared/status/<taskId>.status.json
3. sessions_send 通知對應幕僚
4. 追蹤 shared/results/ 等待結果
5. 彙整回報 James
```

---

## Step 2：設定 openclaw.json

在 `~/.openclaw/openclaw.json` 的 `agents.list` 加入幕僚長：

```json
{
  "id": "dc-channel--<channel_id>",
  "name": "Discord｜幕僚長｜協調中心",
  "workspace": "workspace-<agent-id>",
  "subagents": {
    "allowAgents": [
      "dc-channel--<codex-id>",
      "dc-channel--<test-id>",
      "tg-group--<doc-id>"
    ]
  }
}
```

**關鍵欄位**：
- `workspace`：指定獨立 workspace（與其他 agent 隔離）
- `subagents.allowAgents`：允許派工的幕僚 ID 清單（陣列）

---

## Step 3：建立 shared/ 目錄

```bash
mkdir shared/tasks shared/results shared/status
```

---

## Step 4：建立 AGENTS.md（protocol 文件）

在幕僚長的 workspace 內建立 `AGENTS.md`，定義：

### 4.1 幕僚分工表

```json
{
  "agents": [
    { "id": "dc-channel--...", "name": "Codex 修補手", "role": "實作" },
    { "id": "tg-group--...", "name": "測霸", "role": "測試" },
    { "id": "tg-group--...", "name": "文修", "role": "文檔" }
  ]
}
```

### 4.2 任務格式

```json
{
  "taskId": "2026-03-26_161200_coding",
  "title": "實作功能 X",
  "description": "...",
  "assignee": "dc-channel--...",
  "priority": "high|medium|low",
  "createdBy": "James",
  "createdAt": "2026-03-26T08:12:00Z",
  "successCriteria": ["條件1", "條件2"],
  "status": "pending"
}
```

### 4.3 派工 Protocol

```
Step 1：評估任務（30秒預判）
  → 需要哪個幕僚？
  → 成功標準是什麼？

Step 2：建立任務檔案
  → shared/tasks/<taskId>.json
  → shared/status/<taskId>.status.json

Step 3：通知幕僚
  → sessions_send 發送任務

Step 4：追蹤進度
  → 檢查 shared/results/

Step 5：彙整交付
  → 摘要給 James
  → 更新 status=completed
```

---

## Step 5：建立 MEMORY.md（幕僚長專屬記憶）

```markdown
# MEMORY.md — 幕僚長專用記憶

## 核心 Protocol
1. 派任務時同時寫 tasks/ 和 status/（不可只寫一個）
2. 機密資料不寫入 shared/
3. 同一幕僚同時不超過 3 個任務

## 派工視覺化
James → 理解意圖 → 拆解任務 → 派工 → 追蹤 → 彙整回報
```

---

## Step 6：建立 dispatch_task.py（可選 CLI 工具）

```python
# shared/dispatch_task.py
# 用法：python dispatch_task.py dispatch --title "..." --assignee "..." --desc "..."
# 或直接寫 JSON 檔

import json, os, time
from pathlib import Path

SHARED = Path('~/.openclaw/workspace/shared')

def dispatch(title, assignee, description):
    ts = time.strftime('%Y-%m-%d_%H%M%S')
    task_id = f"{ts}_{title[:20].replace(' ', '_')}"
    
    # 同時寫 tasks/ 和 status/
    task_file = SHARED / 'tasks' / f'{task_id}.json'
    status_file = SHARED / 'status' / f'{task_id}.status.json'
    
    # ... 寫入邏輯
    return task_id
```

---

## 各幕僚的設定

每個幕僚都需要在自己的 workspace 內建立接收任務的 protocol。

### Codex 修補手（dc-channel）

```markdown
## SOUL.md 追加
### 接收任務
- 從 shared/tasks/ 讀取指派給自己的任務
- 完成後結果寫入 shared/results/<taskId>__<agentId>-result.json
- 結果格式：
  {
    "taskId": "...",
    "agentId": "dc-channel--...",
    "status": "success|failed",
    "summary": "30字摘要",
    "output": "詳細輸出",
    "completedAt": "ISO時間"
  }
```

---

## 檔案命名規則

| 類型 | 格式 |
|------|------|
| 任務 | `<YYYY-MM-DD_HHMMSS>_<類型>.json` |
| 狀態 | `<taskId>.status.json` |
| 結果 | `<taskId>__<agentId>-result.json` |

---

## 注意事項

1. **同時寫入原則**：派任務時 `tasks/` 和 `status/` 必須同時寫入，不可只寫一個
2. **狀態同步**：任務完成後要更新 `status/`，否則會被誤判為pending
3. **不要放機密資料**：API key、密碼等嚴禁寫入 shared/
4. **命名一致**：`taskId` 是所有檔案的連接鍵，必須完全一致
5. **Sessions_send**：派工時需要用 sessions_send 即時通知，否則幕僚不會主動讀取

---

## 常見問題

**Q：為什麼不用 Discord 頻道直接派工？**
A：頻道訊息會有汙染問題，且任務追蹤困難。shared/ 方式讓所有交接有結構化記錄。

**Q：幕僚怎麼知道有新任務？**
A：指揮中心用 sessions_send 即時通知。幕僚也可以定期輪詢 shared/tasks/。

**Q：多個幕僚同時處理同一任務？**
A：一個任務只派給一個幕僚。如果需要協作，拆成多個子任務。

---

*本架構參考 vocus.cc 文章「告別混亂！用 AI Agent 工作流建立高效協作環境」設計，並針對 OpenClaw 實作調整。*
