import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import pickle


def generate_final_features(df):
    """为LightGBM生成最终的、包含截面排序的高级特征集"""
    print("开始生成最终版高级特征...")

    # --- 使用您指定的2024年之后的数据窗口 ---
    df['Date'] = pd.to_datetime(df['Date'])
    cutoff_date = pd.to_datetime('2024-01-01')
    df = df[df['Date'] >= cutoff_date].copy()
    print(f"数据已筛选，只使用 {cutoff_date.date()} 之后的数据...")

    df.sort_values(by=['Date', 'StockCode'], inplace=True)

    # --- 核心修复：使用标准的shift方法定义目标变量，确保方向正确 ---
    # 先计算当日收益率，再向上移动一位，得到未来1日的收益率
    df['return_1d'] = df.groupby('StockCode')['Close'].pct_change()
    df['Target'] = df.groupby('StockCode')['return_1d'].shift(-1)

    # --- 特征工程 ---
    # 类别1: 时序特征
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f'return_lag_{lag}'] = df.groupby('StockCode')['return_1d'].shift(lag)

    df['volatility_21'] = df.groupby('StockCode')['return_1d'].transform(lambda x: x.rolling(21).std())
    df['rsi'] = df.groupby('StockCode')['Close'].transform(lambda x: RSIIndicator(close=x, fillna=True).rsi())

    # 类别2: 截面特征 (精髓)
    date_groups = df.groupby('Date')
    df['rank_return_1d'] = date_groups['return_1d'].rank(pct=True)
    df['rank_volatility_21'] = date_groups['volatility_21'].rank(pct=True)
    df['rank_rsi'] = date_groups['rsi'].rank(pct=True)

    # --- 数据清洗 ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 保存完整的特征集，训练时再dropna
    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    combined_df = pd.concat([train_df, test_df]).drop_duplicates(subset=['股票代码', '日期'])
    column_mapping = {"股票代码": "StockCode", "日期": "Date", "收盘": "Close"}
    combined_df = combined_df.rename(columns=column_mapping)

    featured_df = generate_final_features(combined_df)

    final_feature_cols = [col for col in featured_df.columns if col not in ['StockCode', 'Date', 'Target', 'Close']]

    outputdata_path = "./temp/final_featured_data.csv"
    featured_df.to_csv(outputdata_path, index=False)

    with open('./model/final_proc_info.pkl', 'wb') as f:
        pickle.dump({'feature_cols': final_feature_cols}, f)

    print(f"最终特征工程完成，生成 {len(final_feature_cols)} 个特征。数据已保存。")