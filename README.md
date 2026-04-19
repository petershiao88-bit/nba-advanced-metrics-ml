以 **NBA 球隊—球季** 為單位，使用多維進階指標預測**勝率（Win Rate）**。本 repo 以**研究成果分享**為主，整理了資料流程設計、防洩漏前處理、特徵篩選、多模型比較與 SHAP 解釋的完整分析脈絡。

## 重要聲明（資料與授權）

- **本專案不附原始資料**。研究中使用之部分資料來源涉及**付費或授權限制**，不得於公開 repo 散布匯出檔、爬蟲細節或任何可重建該資料之敏感資訊（含帳密、token、cookies、session、headers、私有 API 等）。
- 公開版本僅包含：**管線程式、欄位/schema 說明、研究結果摘要（[`research/RESULTS_SNAPSHOT.md`](research/RESULTS_SNAPSHOT.md)）、評估圖表（[`assets/figures/`](assets/figures/)）**；另附 `sample_data/` 合成資料僅供無授權檔時示範跑通。
- **使用者須自行準備具合法授權之資料**，並依 [`data/README.md`](data/README.md) 與 [`schema.json`](schema.json) 置於本機 `data/processed/`（該目錄已列於 `.gitignore`，不會進入 git）。
- 若 `data/processed/` 為空，執行 `main.py` 會改用 **`sample_data/processed/`** 內之**隨機合成 CSV**（僅供結構與程式可執行性展示，**非真實統計**）。

## 研究 / 預測目標

- **目標變數**：`Win Rate`（API 可傳 `win_rate`，會自動對應欄位）。
- **驗證方式**：依 **年度切分**（`year_holdout` / `rolling_year`）評估，降低時間洩漏。
- **模型管線**：共線性處理 → **LassoCV**、**ElasticNetCV** → **XGBoost**（網格搜尋）→ **SHAP** 與圖表。

## 研究設計摘要

- **研究問題**：球隊層級的多維進階指標，預測 NBA 球季勝率。
- **分析單位**：以「球隊 × 球季」為一筆樣本，將每季資料整理成 `YYYY_metrics.csv`，再合併成跨年度 panel。
- **資料設計**：將目標欄設為 `Win Rate`，其餘數值型進階指標作為候選特徵；檔名中的年份會轉為 `year` 欄，供時間切分使用。
- **方法設計**：先做缺值處理、標準化、低變異與共線性篩選，再比較 **LassoCV**、**ElasticNetCV** 與 **XGBoost** 三條模型路徑。
- **驗證設計**：以年度切分做 holdout / rolling validation，讓所有前處理與特徵選擇都只在訓練年度擬合，避免未來資訊滲入。
- **解釋設計**：除模型分數外，也保留係數圖、殘差圖、特徵縮減圖與 SHAP summary，讓結果不只停留在預測準確度。

## 研究成果摘要（可公開的數字與結論）

- **完整數字表**：見 [`research/RESULTS_SNAPSHOT.md`](research/RESULTS_SNAPSHOT.md)（含測試集 R²、RMSE、holdout 年度與簡短解讀）。此檔可安全提交到 GitHub，**不含**原始 CSV。
- **圖表**：以 [`assets/figures/`](assets/figures/) 為準（與一次完整 `main.py` 管線輸出對齊）。

**本次管線（本機授權資料、year holdout 自動切分）摘要**（詳見上連結）：

- **測試年度**：2025（訓練年度組合見快照檔）。
- **測試集**：LassoCV `R² ≈ 0.85`、`RMSE ≈ 0.062`；ElasticNetCV `R² ≈ 0.85`、`RMSE ≈ 0.062`；XGBoost `R² ≈ 0.79`、`RMSE ≈ 0.074`。
- **解讀**：線性正則化模型略優於本次 XGBoost 設定，顯示在既有特徵與切分下勝率與指標關係偏可線性近似；仍建議對照 SHAP 與係數圖檢查特徵穩定性。

### 如何用「原始資料」重跑並更新 GitHub 展示

資料需在本機（例如 `Team Metric Web Crawler\data\*.csv`）；**請勿**將該資料夾提交（已列於 `.gitignore`）。在**專案根目錄**執行：

```bash
python scripts/run_research_publish.py --source ".\Team Metric Web Crawler\data"
```

若資料夾在別處，改用絕對路徑：

```bash
python scripts/run_research_publish.py --source "C:\Users\你的帳號\...\Team Metric Web Crawler\data"
```

- 會設定 `NBA_PROCESSED_DATA_DIR` 指向該資料夾並跑 `main.py`，終端輸出寫入 `outputs/last_pipeline_run.log`（`*.log` 不納入版本控制），並更新 `research/RESULTS_SNAPSHOT.md` 與 `assets/figures/`。
- 可選：`--copy-to-data-processed` 會複製到 `data/processed/`（仍受 `.gitignore` 保護）。

**建議提交（公開）**：`research/RESULTS_SNAPSHOT.md`、`assets/figures/*.png`。**不要提交** `Team Metric Web Crawler/`、`*.csv` 原始匯出、`cookies.pkl` 等。

## 資料思路（高層，無敏感細節）

1. 在你具合法權利的來源取得「球隊—球季」層級面板。  
2. 依檔名規則 `YYYY_metrics.csv` 分季存放，合併後形成含 `year` 之面板。  
3. 詳見 [`data/README.md`](data/README.md)（表結構、建模 input 對應、本機放置方式）。

## 專案流程

```text
合法資料（本機 data/processed/）或 合成示範（sample_data/processed/）
        → 前處理與共線性篩選（僅在訓練段 fit）
        → LassoCV / ElasticNetCV
        → XGBoost + GridSearchCV
        → 評估與 SHAP / 圖表
```

## 目錄結構

| 路徑 | 說明 |
|------|------|
| `main.py` | 入口：解析資料目錄、執行完整管線、輸出至 `outputs/figures/`。 |
| `src/nba_win_rate_pipeline.py` | 核心管線（Stage 1–3 與繪圖）。 |
| `src/data_layout.py` | 解析 `data/processed/` 或退回 `sample_data/processed/`。 |
| `data/README.md` | **資料政策、表結構與放置說明**。 |
| `schema.json` | 抽象欄位/schema 說明。 |
| `sample_data/processed/` | 合成示範 `*_metrics.csv`（可進 git）。 |
| `scripts/generate_synthetic_sample.py` | 重產示範 CSV（同樣為合成，非真實賽事）。 |
| `scripts/run_research_publish.py` | 用本機 `*_metrics.csv` 跑管線、更新 `research/RESULTS_SNAPSHOT.md` 與 `assets/figures/`。 |
| `research/RESULTS_SNAPSHOT.md` | 可公開的研究數字摘要（不含原始資料）。 |
| `notebooks/metric_selection/` | 特徵探索 notebook（讀取同上資料解析邏輯）。 |
| `assets/figures/` | README 用示意圖（重跑結果以本機為準）。 |
| `outputs/figures/` | 執行管線產出之圖表。 |
| `src/scraping/README.md` | 僅概念說明，**不含**可執行擷取程式。 |

## 使用技術

Python、pandas、NumPy、scikit-learn、XGBoost、SHAP、Matplotlib。

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

## 執行

於專案根目錄：

```bash
python main.py
```

會自動選擇 `data/processed/`（若存在 `*_metrics.csv`）或否則使用 `sample_data/processed/`。

以程式呼叫：

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent  # 依你的檔案調整
sys.path.insert(0, str(ROOT / "src"))
from data_layout import resolve_processed_data_dir
from nba_win_rate_pipeline import load_metrics_from_csvs, run_full_pipeline

csv_dir = resolve_processed_data_dir(ROOT)
df = load_metrics_from_csvs(str(csv_dir))
# run_full_pipeline(df, split_mode="year_holdout", ...)
```

## 研究分享重點

- 資料流程與面板合併設計  
- 特徵工程與共線性 / 正則化路徑  
- 多模型比較與時間切分評估  
- SHAP、係數圖、殘差與回測圖（見 `assets/figures/` 與本機 `outputs/figures/`）  

## 結果圖與解讀

以下為管線可能產出之圖表類型與閱讀方式。圖中數值會隨你使用的資料而改變；目前公開 repo 附圖主要作為研究流程與解釋方式的示意。

| 類型 | 說明 |
|------|------|
| `pipeline_flowchart.png` | 管線階段總覽 |
| `feature_reduction.png` | 特徵數量變化 |
| `model_performance_comparison.png` | 模型測試表現對照 |
| `shap_summary.png` | 特徵貢獻解釋 |

![Pipeline overview](assets/figures/pipeline_flowchart.png)

`pipeline_flowchart.png`：這張圖用來說明研究設計的整體流程，從資料進入模型前的前處理、特徵篩選，到最終模型訓練與解釋步驟。它的重要性在於讓讀者理解本研究不是單純套模型，而是先處理資料品質與特徵冗餘，再進行模型比較。

![Model comparison (example)](assets/figures/model_performance_comparison.png)

`model_performance_comparison.png`：這張圖用來比較不同模型在測試資料上的整體表現。若某一模型表現較好，代表其在目前資料設計下更能捕捉球隊勝率與進階指標之間的關係；若模型差異不大，則表示資料訊號可能已足以讓較簡單模型得到相近結果，也能支持模型選擇上的可解釋性考量。

![SHAP summary (example)](assets/figures/shap_summary.png)

`shap_summary.png`：這張圖用來解釋模型預測背後最重要的特徵。橫軸代表特徵對預測的影響方向與強度，顏色代表特徵值高低。研究上，這類圖能幫助回答「哪些進階指標更容易把球隊勝率往上或往下推動」，讓模型結果不只是準不準，還能被理解與討論。

## 未來可優化方向

- 更多球季與外生變數；機率校準與不確定性量化。  
- 與完全公開之資料來源做穩健性對照（在合規前提下）。  
- 模組化測試與 CI。

## 合規提醒

勿在 issue、PR 或截圖中张贴付費資料、帳密或可識別之私有匯出。若曾誤提交敏感檔，請輪替憑證並考慮清理 Git 歷史。
