import pandas as pd
from automl_pipeline.trainer import Trainer
from sklearn.ensemble import RandomForestClassifier

def test_trainer():
    X = pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
    y = [0,1,0,1]
    t = Trainer(RandomForestClassifier(), test_size=0.5)
    X_train, X_test, y_train, y_test = t.split(X, y)
    t.train(X_train, y_train)
    preds = t.predict(X_test)
    assert len(preds) == len(y_test)