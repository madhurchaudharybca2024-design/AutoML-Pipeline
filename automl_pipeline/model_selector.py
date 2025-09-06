from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def get_model(name: str, params: dict):
    if name == 'random_forest':
        return RandomForestClassifier(**params)
    elif name == 'logistic_regression':
        return LogisticRegression(**params)
    elif name == 'xgboost':
        if XGBClassifier is not None:
            return XGBClassifier(**params)
        else:
            raise ImportError("XGBoost is not installed. Remove it from your config or install via pip.")
    elif name == 'lightgbm':
        if LGBMClassifier is not None:
            return LGBMClassifier(**params)
        else:
            raise ImportError("LightGBM is not installed. Remove it from your config or install via pip.")
    else:
        raise ValueError(f"Unsupported model: {name}")