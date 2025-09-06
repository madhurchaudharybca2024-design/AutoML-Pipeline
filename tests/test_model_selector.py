from automl_pipeline.model_selector import get_model

def test_get_model():
    m = get_model('random_forest', {'n_estimators': 10})
    assert m is not None