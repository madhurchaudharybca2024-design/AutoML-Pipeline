import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Validate required fields
        if 'data_path' not in self.config or 'target' not in self.config or 'models' not in self.config:
            raise ValueError("Config must contain data_path, target, and models fields.")
        self.data_path = self.config['data_path']
        self.target = self.config['target']
        self.models = self.config['models']
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.metrics = self.config.get('metrics', ['accuracy', 'f1'])