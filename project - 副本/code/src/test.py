# code/sre/test.py

import pandas as pd
import numpy as np
import pickle
import os
import glob
from config import *  # 导入所有配置


def final_ensemble_prediction():
    """
    使用所有训练好的模型进行模型融合预测。
    """
    print("--- Step 3: Running Final Ensemble Prediction for April 28 ---")

    # 1. 加载基础特征数据
    base_feature_path = os.path.join(TEMP_DIR, 'base_features.csv')
    df_base = pd.read_csv(base_feature_path)

    # 2. 准备预测输入
    print(f"Preparing input data using a {TIME_WINDOW_SIZE}-day window...")
    prediction_inputs = []
    stock_codes_for_prediction = []

    for code, group in df_base.groupby('StockCode'):
        if len(group) < TIME_WINDOW_SIZE:
            continue

        prediction_window = group.tail(TIME_WINDOW_SIZE)
        flattened_features = []
        for lag in range(1, TIME_WINDOW_SIZE + 1):
            day_data = prediction_window[FLATTEN_COLS].tail(lag).iloc[0]
            flattened_features.extend(day_data.values)

        prediction_inputs.append(flattened_features)
        stock_codes_for_prediction.append(code)

    feature_names = [f'{col}_lag_{lag}' for lag in range(1, TIME_WINDOW_SIZE + 1) for col in FLATTEN_COLS]
    X_predict = pd.DataFrame(prediction_inputs, columns=feature_names)

    # 3. 加载所有模型并进行预测
    all_predictions = []
    model_files = glob.glob(os.path.join(MODEL_DIR, '*.pkl'))

    if not model_files:
        print("Error: No trained models found in `model` directory. Please run `train.py` first.")
        return

    print(f"Found {len(model_files)} models for ensembling...")
    for model_path in model_files:
        print(f"  > Predicting with {os.path.basename(model_path)}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        preds = model.predict(X_predict)
        all_predictions.append(preds)

    # 4. 融合预测 (取平均)
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # 5. 结果展示
    print("\n" + "=" * 60)
    print("      ENSEMBLE PREDICTION RESULTS FOR MONDAY, APRIL 28")
    print("=" * 60)

    results_df = pd.DataFrame({
        'StockCode': stock_codes_for_prediction,
        'Predicted_Change_Pct': ensemble_predictions
    }).sort_values(by='Predicted_Change_Pct', ascending=False)

    print("\n--- Top 10 Stocks with HIGHEST Predicted Price Change ---")
    print(results_df.head(10).to_string(index=False))

    print("\n--- Top 10 Stocks with LOWEST Predicted Price Change ---")
    print(results_df.tail(10).sort_values(by='Predicted_Change_Pct').to_string(index=False))
    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    final_ensemble_prediction()