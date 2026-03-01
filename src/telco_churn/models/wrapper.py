import mlflow
from catboost import CatBoostClassifier
from telco_churn.features.preprocessing import preprocess_features


class CatBoostWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the model from the context.
        """
        self.model = CatBoostClassifier()
        self.model.load_model(context.artifacts["catboost_model"])

    def predict(self, context, model_input):
        """
        Preprocess input and make predictions.
        """
        X, _ = preprocess_features(model_input, target_col="Churn")
        return self.model.predict(X)
