import pandas as pd
import pickle

if __name__ == "__main__":
    print("开始使用最终模型进行预测...")

    with open('./model/final_proc_info.pkl', 'rb') as f:
        proc_info = pickle.load(f)
        feature_cols = proc_info['feature_cols']

    model = pickle.load(open("./model/final_lgbm_model.pkl", 'rb'))

    featured_df = pd.read_csv("./temp/final_featured_data.csv")

    latest_date = featured_df['Date'].max()
    print(f"正在为日期 {latest_date} 的数据，预测下一个交易日的涨跌幅...")
    predict_df = featured_df[featured_df['Date'] == latest_date].copy()

    if len(predict_df) == 0:
        raise ValueError(f"错误：在特征数据中找不到日期为 {latest_date} 的数据用于预测。")

    X_pred = predict_df[feature_cols]

    X_pred.fillna(0, inplace=True)

    predictions = model.predict(X_pred)

    result = pd.DataFrame({
        'StockCode': predict_df['StockCode'],
        'predicted_return': predictions
    })

    result.sort_values(by='predicted_return', ascending=False, inplace=True)

    pred_top_10_max_target = result['StockCode'].head(10).tolist()
    pred_top_10_min_target = result['StockCode'].tail(10).tolist()

    submission = pd.DataFrame({
        "涨幅最大股票代码": pred_top_10_max_target,
        "涨幅最小股票代码": pred_top_10_min_target,
    })

    submission.to_csv("./output/result.csv", index=False, encoding='utf-8')
    print("预测完成，结果已保存到 ./output/result.csv")