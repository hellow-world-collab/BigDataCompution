# code/sre/config.py

# --- 1. 文件路径配置 ---
DATA_DIR = './data/'
TEMP_DIR = './temp/'
MODEL_DIR = './model/'

# --- 2. 特征工程配置 ---
# 用过去多少天的数据来构建一个特征窗口
TIME_WINDOW_SIZE = 32
# 需要展平的原始特征列
FLATTEN_COLS = ['Close', 'Volume', 'TurnoverRate', 'High', 'Low', 'Open', 'daily_amplitude']

# --- 3. 训练与验证配置 ---
# 交叉验证的折叠数
CV_FOLDS = 5
# 每折验证集包含的天数
# CV_FOLDS * VALIDATION_DAYS_PER_FOLD 应该是你希望用来做验证的总天数
VALIDATION_DAYS_PER_FOLD = 10

# --- 4. 超参数优化配置 ---
# Optuna为每个模型在每折CV中尝试优化的次数
OPTUNA_TRIALS = 25  # 设置为0则跳过优化，使用下面的默认参数

# --- 5. 模型配置 ---
# 在这里定义你想要训练的模型和它们的默认参数
# test.py会自动加载所有训练好的模型进行融合

MODELS_TO_TRAIN = {
    'lgbm': {
        'model': 'LGBMRegressor',
        'default_params': {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }
    },
    'xgb': {
        'model': 'XGBRegressor',
        'default_params': {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': 0.02,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'lambda': 1,
            'alpha': 1,
            'max_depth': 5,
            'n_jobs': -1,
            'seed': 42,
        }
    },
    # 您可以在这里添加更多模型，例如 CatBoost
}