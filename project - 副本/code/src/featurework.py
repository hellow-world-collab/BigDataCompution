# code/sre/feature.py

import pandas as pd
import numpy as np
import os
import warnings
from config import DATA_DIR, TEMP_DIR  # 从config导入路径

warnings.filterwarnings('ignore')


def ewma(series, span):
    """计算指数移动平均线 (EMA)"""
    return series.ewm(span=span, adjust=False).mean()


def create_base_features():
    """
    最终版特征工程脚本，包含所有基础、截面、技术指标和高阶统计特征。
    """
    print("--- Step 1: Running Ultimate Feature Engineering ---")

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

    # 2. 日期和基础衍生特征
    df["Date"] = pd.to_datetime(df["Date"])
    df['date_id'] = (df['Date'] - df['Date'].min()).dt.days
    df['daily_amplitude'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6)

    # 按照股票和日期排序，为后续计算做准备
    df = df.sort_values(by=['StockCode', 'date_id']).reset_index(drop=True)

    # ================================================================= #
    # 3. 新增：丰富的技术分析指标
    # ================================================================= #
    print("Generating technical analysis indicators (MACD, RSI, Bollinger, ATR)...")
    grouped_by_stock = df.groupby('StockCode')

    # MACD
    ema_12 = grouped_by_stock['Close'].transform(lambda x: ewma(x, 12))
    ema_26 = grouped_by_stock['Close'].transform(lambda x: ewma(x, 26))
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = grouped_by_stock['macd'].transform(lambda x: ewma(x, 9))
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = grouped_by_stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma_20 = grouped_by_stock['Close'].transform(lambda x: x.rolling(window=20).mean())
    std_20 = grouped_by_stock['Close'].transform(lambda x: x.rolling(window=20).std())
    df['bollinger_upper'] = ma_20 + (std_20 * 2)
    df['bollinger_lower'] = ma_20 - (std_20 * 2)
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / (ma_20 + 1e-6)
    df['bollinger_percent_b'] = (df['Close'] - df['bollinger_lower']) / (
                df['bollinger_upper'] - df['bollinger_lower'] + 1e-6)

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - grouped_by_stock['Close'].shift())
    low_close = np.abs(df['Low'] - grouped_by_stock['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = grouped_by_stock.apply(lambda x: true_range[x.index].rolling(window=14).mean()).reset_index(0,
                                                                                                            drop=True)

    # ================================================================= #
    # 4. 新增：高阶统计特征
    # ================================================================= #
    print("Generating higher-order statistical features (skew, kurtosis)...")
    daily_return = grouped_by_stock['Close'].pct_change()
    df['rolling_skew_20'] = grouped_by_stock.apply(
        lambda x: daily_return[x.index].rolling(window=20).skew()).reset_index(0, drop=True)
    df['rolling_kurt_20'] = grouped_by_stock.apply(
        lambda x: daily_return[x.index].rolling(window=20).kurt()).reset_index(0, drop=True)

    # ================================================================= #
    # 5. 截面特征 (Cross-sectional Features)
    # ================================================================= #
    print("Generating cross-sectional (rank) features...")
    grouped_by_date = df.groupby('date_id')

    # 对所有我们认为重要的指标进行截面排序
    rank_cols = [
        'Close', 'Volume', 'TurnoverRate', 'daily_amplitude', 'label',
        'macd', 'rsi', 'bollinger_width', 'atr'
    ]
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_rank'] = grouped_by_date[col].rank(pct=True)

    # 6. 清理并保存最终数据
    print("Cleaning and saving final feature set...")
    # 用0填充所有在计算过程中产生的NaN值
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    os.makedirs(TEMP_DIR, exist_ok=True)
    output_path = os.path.join(TEMP_DIR, 'base_features.csv')
    df.to_csv(output_path, index=False)

    print(f"--- Ultimate features saved to {output_path} ---")


if __name__ == '__main__':
    create_base_features()