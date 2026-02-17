from farm_detection.models.predict import Predictor

def test_predict():
    predictor = Predictor(model_path="./model/gaussiannb.joblib", preprocessor_path="./model/preprocessor.joblib")
    test_data = [[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362]] 
    class_pred, decoded_pred = predictor.predict(test_data)
    assert class_pred == [17]
    assert decoded_pred == ['papaya']
