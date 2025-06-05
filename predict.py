import joblib
from tensorflow.keras.models import load_model

scaler = joblib.load("scaler.pkl")
model = load_model("model_weights.h5")

def predict_score(df1, df2):
    combined = pd.concat([df1, df2], axis=1)
    scaled = scaler.transform(combined)
    prediction = model.predict(scaled)
    return prediction[0]
