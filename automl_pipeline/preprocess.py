from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="mean")

    def fit_transform(self, X: pd.DataFrame):
        X = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])
        if not X_num.empty:
            X[X_num.columns] = self.imputer.fit_transform(X_num)
            X[X_num.columns] = self.scaler.fit_transform(X[X_num.columns])
        for col in X_cat.columns:
            le = LabelEncoder()
            # Fill NaNs with a placeholder
            X[col] = le.fit_transform(X[col].astype(str).fillna("UNK"))
            self.label_encoders[col] = le
        return X

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])
        if not X_num.empty:
            X[X_num.columns] = self.imputer.transform(X_num)
            X[X_num.columns] = self.scaler.transform(X[X_num.columns])
        for col in X_cat.columns:
            le = self.label_encoders.get(col)
            if le is not None:
                # Handle unseen labels
                vals = X[col].astype(str).fillna("UNK")
                vals = vals.apply(lambda x: x if x in le.classes_ else "UNK")
                X[col] = le.transform(vals)
        return X