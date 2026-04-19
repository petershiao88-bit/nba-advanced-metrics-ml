# Synthetic demo data

此目錄下的 `processed/*_metrics.csv` 為 **隨機合成、僅供結構展示與管線可執行**，不代表真實 NBA 統計，也**不可**用於研究結論。

公開專案不附任何付費或授權資料；請在你取得合法授權後，將符合 [`schema.json`](../schema.json) 的 CSV 置於 `data/processed/`。

`processed/*.csv` 內之 `Win Rate` 與 `metric_*` 為**隨機合成並加上弱人造訊號**，僅讓管線可跑通、**不可**解讀為真實賽事結論。若要重產相同種子之示範檔，可自專案根目錄執行：`python scripts/generate_synthetic_sample.py`。
