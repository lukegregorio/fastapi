from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

import joblib

# Load dataset
data = load_iris()
data = pd.DataFrame(data.data, columns=data.feature_names)
data['species'] = load_iris().target

# Train model
model = RandomForestClassifier()
model.fit(data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']], data['species'])

# Save model
joblib.dump(model, 'model.joblib')