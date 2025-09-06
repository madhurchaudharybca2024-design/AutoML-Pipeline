from sklearn.metrics import accuracy_score, f1_score

class Evaluator:
    def __init__(self, metrics=['accuracy', 'f1']):
        self.metrics = metrics

    def evaluate(self, y_true, y_pred):
        results = {}
        if 'accuracy' in self.metrics:
            try:
                results['accuracy'] = accuracy_score(y_true, y_pred)
            except Exception as e:
                results['accuracy'] = f"Error computing accuracy: {str(e)}"
        if 'f1' in self.metrics:
            try:
                results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            except Exception as e:
                results['f1'] = f"Error computing f1: {str(e)}"
        return results