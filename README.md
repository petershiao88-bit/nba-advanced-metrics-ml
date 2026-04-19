# NBA Team Win Rate Prediction
## 以進階球隊指標結合機器學習預測 NBA 勝率

本專案以 **NBA 球隊—球季（team-season）** 為分析單位，利用多項進階指標建立機器學習模型，預測球隊單季 **勝率（Win Rate）**。  
公開版本重點放在 **資料流程設計、避免資料洩漏的前處理、特徵篩選、多模型比較，以及模型解釋性分析**，適合作為資料分析與機器學習作品集展示。

---

## 專案亮點

- 建立以 **球隊—球季面板資料** 為基礎的預測流程
- 以 **年度切分** 進行驗證，降低時間洩漏風險
- 結合 **LassoCV、ElasticNetCV、XGBoost** 進行模型比較
- 使用 **SHAP** 提供模型可解釋性分析
- 提供 **合成示範資料**，讓公開版本可執行且不涉及授權風險

---

## 資料與授權聲明

### Data Availability

本專案**不提供原始研究資料**。研究中使用的部分資料來源涉及 **付費或授權限制**，因此公開版本不包含：

- 原始資料檔
- 匯出後的付費資料
- 爬蟲實作細節
- 帳號、密碼、token、cookies、session、headers
- 私有 API 或任何可直接重建資料來源的敏感資訊

本 repo 公開內容僅包含：

- 模型與資料處理管線
- 合成示範資料
- 資料結構與欄位說明
- 圖表範例與分析流程

若你擁有**合法授權的資料來源**，可依照 [`data/README.md`](data/README.md) 與 [`schema.json`](schema.json) 的說明，將資料放置於本機 `data/processed/` 後重現分析流程。

若 `data/processed/` 內未放入合法資料，系統會自動改讀取 `sample_data/processed/` 中的**合成資料**，僅供流程展示與程式測試使用，**不代表真實球隊表現**。

---

## 研究目標

本專案的主要目標為：

- 使用 NBA 球隊進階指標預測球季勝率
- 比較不同模型在年度切分情境下的表現
- 分析哪些特徵對勝率預測最具影響力
- 建立具備可讀性與可重現性的 ML workflow

### Target Variable
- `Win Rate`
- 若使用程式呼叫，也可傳入 `win_rate`，系統會自動對應欄位名稱

---

## 方法概覽

整體流程如下：

```text
合法授權資料（data/processed/）
        或
合成示範資料（sample_data/processed/）
        ↓
資料合併與前處理
        ↓
共線性處理與特徵篩選
        ↓
LassoCV / ElasticNetCV
        ↓
XGBoost + GridSearchCV
        ↓
模型評估、圖表輸出與 SHAP 解釋
