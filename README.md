# Wine Quality Regression

> HW2｜Multiple Linear Regression｜CRISP-DM Analysis

本專案使用 Kaggle Wine Quality Dataset，依照 CRISP-DM 完成資料理解、資料準備、特徵選擇、線性回歸建模與模型評估。研究目標是根據葡萄酒的理化測量值，預測感官品質分數 `quality`。

## Research Snapshot

| 項目 | 內容 |
|---|---|
| 資料來源 | [Kaggle Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) |
| 原始研究來源 | [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| 實際資料量 | 1,143 筆、11 個輸入特徵 |
| 目標變數 | `quality`，葡萄酒品質分數 |
| 資料切分 | 80% training / 20% testing |
| 隨機種子 | `random_state=42` |
| 最佳模型 | 11-feature Multiple Linear Regression |
| 最佳 RMSE | `0.6165` |
| 最佳 R² | `0.3171` |

### Kaggle 名次說明

本資料來源是 Kaggle **Dataset 頁面**，不是 Kaggle Competition，因此沒有官方 leaderboard 或競賽名次。報告中以 `N/A` 正確呈現，不將 Kaggle views、downloads 或 Usability 分數誤稱為競賽排名。

## CRISP-DM Workflow

### 1. Business Understanding

酒商或品管人員希望在感官品評前，先透過固定酸度、揮發性酸度、酒精濃度等可量化化學特徵，估計葡萄酒品質。模型可作為品質篩選與分析輔助，不取代專業品評。

### 2. Data Understanding

資料包含 11 個理化輸入特徵與 1 個品質輸出欄位。資料集中品質分數主要集中於 5 與 6，且各分數樣本數不平衡；這是模型 R² 受限的重要背景。`Id` 僅為識別欄位，不納入預測。

### 3. Data Preparation

- 使用 `kagglehub.dataset_download()` 自動下載資料。
- 移除 `Id` 識別欄位。
- 檢查欄位型態、缺失值、描述統計與品質分布。
- 以固定種子切分 training/testing data。
- 缺失值補值與 SelectKBest 特徵選擇均在 training pipeline 中處理，以避免資料洩漏。

### 4. Modeling

比較三種模型：

1. **A：Simple Linear Regression**，只使用 `alcohol`。
2. **B：Multiple Linear Regression**，使用全部 11 個特徵。
3. **C：Feature Selection + Linear Regression**，用 `SelectKBest(f_regression)` 選出前 5 個特徵。

### 5. Evaluation

使用測試集計算 MAE、MSE、RMSE 與 R²，並以實際值/預測值散佈圖及 95% prediction interval 檢查模型的不確定性。模型 C 選出的特徵為 `volatile_acidity`、`citric_acid`、`density`、`sulphates`、`alcohol`。

### 6. Deployment

模型 Pipeline 以 Joblib 保存，可在本機、Google Colab 或後續的 Streamlit/FastAPI 應用中載入。部署時應同步記錄特徵順序、資料版本、模型版本與新資料的預測誤差。

## Model Results

| 模型 | 特徵數 | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| A：alcohol baseline | 1 | 0.5265 | 0.4175 | 0.6462 | 0.2497 |
| B：全部特徵 | 11 | **0.4773** | **0.3800** | **0.6165** | **0.3171** |
| C：SelectKBest top 5 | 5 | 0.4834 | 0.3856 | 0.6210 | 0.3071 |

全部特徵模型相較 alcohol baseline 的 RMSE 約下降 4.6%。SelectKBest 僅使用 5 個特徵，仍維持接近全部特徵模型的表現，顯示降低輸入維度後仍有實用性。模型 C 的 95% prediction interval 涵蓋率為 `0.9520`，平均區間寬度為 `2.5916`。

## Visual Results

### Feature Correlation

![Feature correlation heatmap](wine_quality_outputs/01_correlation_heatmap.png)

### Prediction with 95% Prediction Interval

![Prediction interval](wine_quality_outputs/03_prediction_interval.png)

### Actual vs Predicted

![Actual versus predicted](wine_quality_outputs/04_actual_vs_predicted.png)

### Model Comparison

![Comparison of three regression models](wine_quality_outputs/06_model_comparison.png)

### RMSE by Number of Features

![RMSE by number of features](wine_quality_outputs/07_rmse_by_features.png)

### R-squared by Number of Features

![R-squared by number of features](wine_quality_outputs/08_r2_by_features.png)

## Reproducibility

### Google Colab

上傳 `5115056030_hw2.py` 或 `5115056030_hw2.ipynb` 後執行：

```python
!pip install -r requirements.txt
!python 5115056030_hw2.py
```

程式會自動從 KaggleHub 下載資料，並將結果寫入 `wine_quality_outputs/`。也可以直接開啟 [5115056030_hw2.ipynb](5115056030_hw2.ipynb) 逐 cell 執行。

### Local Python 3.12

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe 5115056030_hw2.py
```

## Output Files

| 檔案 | 用途 |
|---|---|
| [5115056030_hw2.py](5115056030_hw2.py) | 完整分析主程式 |
| [5115056030_hw2.ipynb](5115056030_hw2.ipynb) | Colab/Jupyter Notebook |
| [5115056030_完整報告.pdf](5115056030_完整報告.pdf) | 正式報告與美化對話摘要合併版 |
| `02_model_metrics.csv` | 模型評估指標 |
| `03_prediction_interval.png` | 預測值與 95% 預測區間 |
| `04_actual_vs_predicted.png` | 實際值與預測值圖 |
| `*.joblib` | 訓練完成的模型 Pipeline |
| [5115056030_hw2.zip](5115056030_hw2.zip) | 僅包含完整報告的繳交壓縮檔 |

## AI-Assisted Research

`5115056030_完整報告.pdf` 已包含 **GPT 輔助內容** 與 **NotebookLM 摘要**，並說明模型數字皆由程式實際執行產生；ZIP 不再重複放置獨立 GPT 對話 PDF。

## Limitations

品質分數是離散且有序的感官評分，並非連續測量值；資料分布也不均衡。因此線性回歸適合做為可解釋的基準模型，但不應直接視為最終品質判定工具。後續可使用交叉驗證比較 Ridge、Random Forest、Gradient Boosting 或 XGBoost，並進一步檢查殘差的線性、常態與同質變異假設。
