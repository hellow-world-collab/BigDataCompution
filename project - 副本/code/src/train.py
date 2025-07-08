# code/sre/train.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import optuna
from sklearn.model_selection import GroupKFold
from config import *  # 导入所有配置


def create_flattened_features(df, window_size, feature_cols):
    """将时序数据展平"""
    lagged_dfs = []
    for lag in range(1, window_size + 1):
        shifted = df[feature_cols].shift(lag)
        shifted.columns = [f'{col}_lag_{lag}' for col in feature_cols]
        lagged_dfs.append(shifted)

    flattened_df = pd.concat(lagged_dfs, axis=1)
    final_df = pd.concat([df[['StockCode', 'date_id', 'label']], flattened_df], axis=1)
    return final_df


def objective(trial, model_name, X_train, y_train, X_val, y_val):
    """Optuna的目标函数"""
    if model_name == 'lgbm':
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-2, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-2, 10.0, log=True),
            'verbose': -1, 'n_jobs': -1, 'seed': 42,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    elif model_name == 'xgb':
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'mae', 'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'n_jobs': -1, 'seed': 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    preds = model.predict(X_val)
    mae = np.mean(np.abs(preds - y_val))
    return mae


def train_all_models():
    """主训练函数"""
    print("--- Step 2: Running Ultimate Model Training ---")

    # 1. 加载基础特征
    base_feature_path = os.path.join(TEMP_DIR, 'base_features.csv')
    df_base = pd.read_csv(base_feature_path)

    # 2. 创建展平特征
    print(f"Creating flattened features with window size {TIME_WINDOW_SIZE}...")
    all_stocks_flattened = [create_flattened_features(group, TIME_WINDOW_SIZE, FLATTEN_COLS) for _, group in
                            df_base.groupby('StockCode')]
    df_model = pd.concat(all_stocks_flattened).dropna()

    # 3. 交叉验证与模型训练循环
    features = [col for col in df_model.columns if col not in ['StockCode', 'date_id', 'label']]
    X, y = df_model[features], df_model['label']

    # 使用GroupKFold确保同一天的所有股票都在同一个折叠里，防止数据泄露
    gkf = GroupKFold(n_splits=CV_FOLDS)

    for model_name, config in MODELS_TO_TRAIN.items():
        print(f"\n--- Training model: {model_name} ---")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df_model['date_id'])):
            print(f"--- Fold {fold + 1}/{CV_FOLDS} ---")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            best_params = config['default_params']
            best_iteration = 2000

            # 4. 超参数优化 (如果开启)
            if OPTUNA_TRIALS > 0:
                print(f"Running Optuna HPO for {OPTUNA_TRIALS} trials...")
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_val, y_val),
                               n_trials=OPTUNA_TRIALS)
                best_params.update(study.best_params)
                print(f"Best MAE for fold {fold + 1}: {study.best_value:.5f}")

            # 5. 使用最佳参数重新训练并保存模型
            print("Retraining model with best parameters on fold data...")
            if model_name == 'lgbm':
                model = lgb.LGBMRegressor(**best_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                best_iteration = model.best_iteration_
            elif model_name == 'xgb':
                model = xgb.XGBRegressor(**best_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
                best_iteration = model.best_iteration

            # 为了预测时使用，我们用完整的fold数据重新训练
            model.set_params(n_estimators=best_iteration)
            model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, f'{model_name}_fold_{fold}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {model_path}")


if __name__ == '__main__':
    train_all_models()