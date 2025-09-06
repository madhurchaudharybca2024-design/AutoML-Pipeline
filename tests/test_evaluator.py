from automl_pipeline.evaluator import Evaluator

def test_evaluator():
    y_true = [0,1,1,0]
    y_pred = [0,1,0,1]
    ev = Evaluator(['accuracy','f1'])
    res = ev.evaluate(y_true, y_pred)
    assert 'accuracy' in res