import uvicorn
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()


class Flower(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    return {"Predict": "Iris species"}

@app.post("/predict")
def predict(flower: Flower):

    model = joblib.load('model.joblib')

    input_data = pd.DataFrame(
        [[flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]],
        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    )

    prediction = model.predict(input_data)

    # Cast prediction to a native Python type (e.g., int or float)
    prediction_result = prediction.item()  # Converts numpy.int64 to native int

    return {"species": prediction_result}


# NEXT STEPS: 
# Lets deploy in prod with heroku. 
# Just to see how to do it like. 
# Then see how I can use streamlit to access my API.
