import mlflow

def train_model(X, y, config):
    raise NotImplementedError

def log_experiment():
    mlflow.log_param("test", 1)
