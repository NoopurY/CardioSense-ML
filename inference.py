import joblib
import numpy as np

model = joblib.load("model.pkl")


def predict(features: list[float]):
    X = np.array(features, dtype=float).reshape(1, -1)
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0].tolist()
    return {"class_id": int(pred), "probabilities": probs}


if __name__ == "__main__":
    demo = [0.1] * 12
    print(predict(demo))
