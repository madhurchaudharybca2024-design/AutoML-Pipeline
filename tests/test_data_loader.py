from automl_pipeline.data_loader import DataLoader

def test_load():
    loader = DataLoader('examples/sample_data.csv')
    df = loader.load()
    assert not df.empty
    assert 'feature1' in df.columns