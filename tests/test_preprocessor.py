from farm_detection.data.preprocess import Preprocessor
import pandas as pd

def test_preprocessor():
    # Instantiate the Preprocessor
    preprocessor = Preprocessor()
    # Create a sample DataFrame to test the fit_transform method
    test_data = [[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362, "rice"]] 
    columns = ["N","P","K","temperature","humidity","ph","rainfall","label"]
    test_df = pd.DataFrame(test_data, columns=columns)
    X_scaled, _ = preprocessor.fit_transform(test_df.drop("label", axis=1), test_df["label"])
    assert X_scaled.dtypes.tolist() == [float, float, float, float, float, float, float]

def test_preprocessor_transform():
    preprocessor = Preprocessor()
    test_data = [[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362, "rice"]] 
    columns = ["N","P","K","temperature","humidity","ph","rainfall","label"]
    test_df = pd.DataFrame(test_data, columns=columns)
    preprocessor.fit(test_df.drop("label", axis=1), test_df["label"])
    X_scaled, _ = preprocessor.transform(test_df.drop("label", axis=1), test_df["label"])
    assert X_scaled.dtype == float

