# 資料蒐集（公開 repo 不附實作）

公開版本**不包含**任何可自動化存取付費或訂閱資料之程式碼、帳密、token、cookies、headers 或私有 endpoint。

高層思路（僅敘事，不可據此繞過授權）：

1. 在你具合法權利的來源取得「球隊—球季」層級的進階指標。  
2. 依 [`schema.json`](../../schema.json) 與 [`data/README.md`](../../data/README.md) 整理欄位與檔名（`YYYY_metrics.csv`）。  
3. 將檔案置於本機 `data/processed/`（已列於 `.gitignore`，不進 git）。  

展示重點為 **資料流程設計、特徵工程、建模與評估**，而非資料來源本身。
