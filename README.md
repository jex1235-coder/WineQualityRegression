# 7114056XXX HW2 - Wine Quality Regression

## Google Colab

```python
!pip install -r requirements.txt
!python 7114056XXX_hw2.py
```

程式會使用 `kagglehub` 下載 `yasserh/wine-quality-dataset`，並在 `wine_quality_outputs/` 產生：

- `02_model_metrics.csv`: 三個線性回歸模型的 MAE、MSE、RMSE、R2
- `03_prediction_interval.png`: SelectKBest 模型的 95% 預測區間
- `04_actual_vs_predicted.png`: 實際值與預測值
- `01_correlation_heatmap.png`: 特徵相關係數熱圖
- `05_prediction_interval_summary.csv`: 預測區間涵蓋率與平均寬度

本次固定使用 `random_state=42`，資料切分為 80% 訓練集與 20% 測試集。

# WineQualityRegression
