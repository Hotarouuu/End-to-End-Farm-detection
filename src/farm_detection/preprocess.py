from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def fit_transform(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        return X_scaled, y_encoded

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform_labels(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)
    
