from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from farm_detection.models.model import GNB
from farm_detection.data.preprocess import Preprocessor
import joblib
import yaml
import mlflow

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train():


    config = load_config("config/model1.yaml")

    # Enable autologging


    mlflow.set_experiment("Naive Bayes Experiment")
    mlflow.sklearn.autolog()
    with mlflow.start_run():

        df = pd.read_csv(config["data"]["train_path"])

        processing = Preprocessor()

        X_scaled, y_encoded = processing.fit_transform(df[config["data"]["features"]], df[config["data"]["target"]])

        print(X_scaled)
    
        train_X, test_X, train_y, test_y = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        model = GNB(priors=config["model"]["variables"]["priors"], var_smoothing=config["model"]["variables"]["var_smoothing"])

        model.fit(train_X, train_y)

        pred = model.predict(test_X)
        print(classification_report(test_y, pred, digits=4))

        preprocessor = {
            'scaler': processing.scaler,
            'labelencoder' : processing.label_encoder}

        joblib.dump(preprocessor, config["artifacts"]["preprocessor_path"])
        joblib.dump(model, config["artifacts"]["model_path"])

        mlflow.log_artifact("config/model1.yaml")
        mlflow.log_artifact(config["artifacts"]["preprocessor_path"])

        print("Model saved.")


if __name__ == "__main__":
    train()