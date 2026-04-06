import joblib
import numpy as np
from sklearn.metrics import classification_report

model = joblib.load("model.pkl")
X = np.random.rand(200, 12)
y = np.random.randint(0, 5, size=200)
yp = model.predict(X)
print(classification_report(y, yp))
