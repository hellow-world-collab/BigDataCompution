import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from scipy.stats import spearmanr


def competition_score_metric(y_true, y_pred):
    """自定义LightGBM评估函数，计算比赛的最终得分"""
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).reset_index(drop=True)

    df.sort_values(by='pred', ascending=False, inplace=True)
    pred_top10 = set(df.index[:10])
    pred_bottom10 = set(df.index[-10:])

    df.sort_values(by='true', ascending=False, inplace=True)
    true_top10 = set(df.index[:10])
    true_bottom10 = set(df.index[-10:])

    precision_up = len(pred_top10.intersection(true_top10)) / 10
    f1_up = 2 * precision_up ** 2 / (2 * precision_up) if precision_up > 0 else 0

    precision_down = len(pred_bottom10.intersection(true_bottom10)) / 10
    f1_down = 2 * precision_down ** 2 / (2 * precision_down) if precision_down > 0 else 0

    common_top = list(pred_top10.intersection(true_top10))
    rank_corr_up = 0
    if len(common_top) > 1:
        pred_rank_up = df.loc[common_top].sort_values(by='pred', ascending=False).index.to_series().rank()
        true_rank_up = df.loc[common_top].sort_values(by='true', ascending=False).index.to_series().rank()
        corr, _ = spearmanr(pred_rank_up, true_rank_up)
        rank_corr_up = corr if not np.isnan(corr) else 0

    common_bottom = list(pred_bottom10.intersection(true_bottom10))
    rank_corr_down = 0
    if len(common_bottom) > 1:
        pred_rank_down = df.loc[common_bottom].sort_values(by='pred', ascending=True).index.to_series().rank()
        true_rank_down = df.loc[common_bottom].sort_values(by='true', ascending=True).index.to_series().rank()
        corr, _ = spearmanr(pred_rank_down, true_rank_down)
        rank_corr_down = corr if not np.isnan(corr) else 0

    final_score = 0.2 * f1_up + 0.2 * f1_down + 0.3 * rank_corr_up + 0.3 * rank_corr_down
    return 'competition_score', final_score, True


if __name__ == "__main__":
    print("开始使用LightGBM模型进行训练（使用最终比赛评分标准进行验证）...")

    with open('./model/final_proc_info.pkl', 'rb') as f:
        proc_info = pickle.load(f)
        feature_cols = proc_info['feature_cols']

    data = pd.read_csv("./temp/final_featured_data.csv")

    # 关键：只使用有Target的数据进行训练和验证
    trainable_data = data.dropna(subset=['Target']).copy()

    # 黄金验证策略
    val_start_date = '2025-04-21'
    train_set = trainable_data[trainable_data['Date'] < val_start_date]
    val_set = trainable_data[trainable_data['Date'] >= val_start_date]

    X_train, y_train = train_set[feature_cols], train_set['Target']
    X_val, y_val = val_set[feature_cols], val_set['Target']

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")

    params = {'objective': 'regression_l1', 'n_estimators': 2000, 'learning_rate': 0.01,
              'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'num_leaves': 16,
              'verbose': -1, 'n_jobs': -1, 'seed': 42}

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric=competition_score_metric,
              callbacks=[lgb.early_stopping(100, verbose=True)])

    print("\n在全部可训练数据上重新训练最终模型...")
    best_iteration = model.best_iteration_ if model.best_iteration_ else 200
    final_params = params.copy()
    final_params['n_estimators'] = best_iteration

    final_train_data = trainable_data[trainable_data['Date'] < '2025-04-25']
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(final_train_data[feature_cols], final_train_data['Target'])

    model_path = "./model/final_lgbm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    print(f"最终模型训练完成（使用 {best_iteration} 轮迭代），已保存到 {model_path}")