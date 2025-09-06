import pandas as pd
from automl_pipeline.preprocess import Preprocessor

def test_preprocessor():
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    pre = Preprocessor()
    out = pre.fit_transform(df)
    assert out.shape == df.shape