# code/sre/feature.py
# (此文件与上一版基本一致，主要是为了确认其职责)
# (您可以继续使用上一版提供的 feature.py)

import pandas as pd
import numpy as np
import os
import warnings
from config import DATA_DIR, TEMP_DIR  # 从config导入路径

warnings.filterwarnings('ignore')


def create_base_features():
    print("--- Step 1: Running Base Feature Engineering ---")

    # 1. 加载和重命名
    train_path = os.path.join(DATA_DIR, 'train.csv')
    df = pd.read_csv(train_path)

    column_mapping = {
        "股票代码": "StockCode", "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "TurnoverValue",
        "振幅": "Amplitude", "涨跌额": "PriceChange", "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    }
    df.rename(columns=column_mapping, inplace=True)
    df.rename(columns={'PriceChangePercentage': 'label'}, inplace=True)

    # 2. 日期处理
    df["Date"] = pd.to_datetime(df["Date"])
    df['date_id'] = (df['Date'] - df['Date'].min()).dt.days
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.weekday

    # 3. 基础衍生特征
    df['daily_amplitude'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6)

    # 4. 保存到temp文件夹
    os.makedirs(TEMP_DIR, exist_ok=True)
    output_path = os.path.join(TEMP_DIR, 'base_features.csv')
    df.to_csv(output_path, index=False)

    print(f"--- Base features saved to {output_path} ---")


if __name__ == '__main__':
    create_base_features()