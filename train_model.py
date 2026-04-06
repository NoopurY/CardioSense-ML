import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Replace with MIT-BIH feature matrix loading pipeline
X = np.random.rand(5000, 12)
y = np.random.randint(0, 5, size=5000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X_res, y_res)

print("Validation score:", model.score(X_test, y_test))
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
