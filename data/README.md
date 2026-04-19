# 資料配置（公開 repo 政策）

## 授權聲明

本專案**不附**任何來自付費或需訂閱之第三方平台的原始匯出檔（csv、xlsx、json 等）。若你曾在本機使用該類資料完成研究，請自行保留備份；**請勿**將該類檔案提交至公開 GitHub。

此 repo 展示的是：

- 資料如何整理成建模用面板（檔名規則、`year` 合併）  
- 前處理、特徵工程、模型訓練、評估與解釋（SHAP 等）  

而非資料來源本身或繞過授權的取得方式。

## 你需要準備什麼（合法取得後）

在 **`data/processed/`**（此路徑下檔案已列於 `.gitignore`，不會進版本庫）放置：

- 檔名：`YYYY_metrics.csv`（例如 `2014_metrics.csv`）  
- 內容：每檔一個球季，**每列一隊**，至少包含：
  - `Team`：隊名或代碼  
  - `Win Rate`：該季勝率（0–1）  
  - 若干**數值型**特徵欄（進階指標等；實際欄名可依你的來源）  

`load_metrics_from_csvs()` 會依檔名數字推斷 `year` 欄並合併多季。

## 與建模 input 的對應

1. 讀取所有 `*_metrics.csv` → 合併為單一 DataFrame，含 `year`。  
2. 目標欄：`Win Rate`（或 `win_rate`，管線會對應）。  
3. 特徵：數值欄位扣除 id / 年份 / 目標（見 `infer_feature_columns`）。  

抽象欄位結構見專案根目錄 **`schema.json`**。

## 示範資料

若 `data/processed/` 為空，執行 `main.py` 會改用 **`sample_data/processed/`** 內的合成 CSV（欄位名為 `metric_*`，**非真實統計**），僅供驗證程式可跑通。

## `data/raw/`

可選：放置你自行轉成 `processed` 前的中間檔；同樣應遵守授權，且預設已列入 `.gitignore` 模式，避免誤提交。
