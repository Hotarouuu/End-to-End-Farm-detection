import joblib

class Predictor:
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, X):
        X_scaled, _ = self.preprocessor['scaler'].transform(X), None
        class_pred = self.model.predict(X_scaled)
        decoded_pred = self.preprocessor['labelencoder'].inverse_transform(class_pred)
        return class_pred, decoded_pred
    
