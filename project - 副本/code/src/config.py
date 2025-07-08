# code/sre/config.py

# --- 1. 文件路径配置 ---
DATA_DIR = './data/'
TEMP_DIR = './temp/'
MODEL_DIR = './model/'

# --- 2. 特征工程配置 ---
TIME_WINDOW_SIZE = 32
# ===================================================================== #
#  需要展平的特征列 (终极完整版：加入了所有技术指标和高阶特征)
# ===================================================================== #
FLATTEN_COLS = [
    # 基础价量特征
    'Close', 'Volume', 'TurnoverRate', 'High', 'Low', 'Open', 'daily_amplitude',

    # 技术指标
    'macd', 'macd_signal', 'macd_hist', 'rsi', 'bollinger_width', 'bollinger_percent_b', 'atr',

    # 高阶统计特征
    'rolling_skew_20', 'rolling_kurt_20',

    # 截面特征 (非常重要!)
    'Close_rank', 'Volume_rank', 'TurnoverRate_rank', 'daily_amplitude_rank', 'label_rank',
    'macd_rank', 'rsi_rank', 'bollinger_width_rank', 'atr_rank'
]

# --- 3. 训练与验证配置 ---
CV_FOLDS = 5
VALIDATION_DAYS_PER_FOLD = 10

# --- 4. 超参数优化配置 ---
OPTUNA_TRIALS = 25  # 设置为0则跳过优化

# --- 5. 模型配置 ---
MODELS_TO_TRAIN = {
    'lgbm': {
        'model': 'LGBMRegressor',
        'default_params': {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
            'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31, 'verbose': -1,
            'n_jobs': -1, 'seed': 42,
        }
    },
    'xgb': {
        'model': 'XGBRegressor',
        'default_params': {
            'objective': 'reg:squarederror', 'eval_metric': 'mae', 'n_estimators': 2000,
            'learning_rate': 0.02, 'colsample_bytree': 0.8, 'subsample': 0.8,
            'lambda': 1, 'alpha': 1, 'max_depth': 5, 'n_jobs': -1, 'seed': 42,
        }
    },
}