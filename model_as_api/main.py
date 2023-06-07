import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
file_name = 'cars_best_pipe.pkl'
with open(file_name, 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


class Prediction(BaseModel):
    id: int
    prediction: str
    price: int


@app.get('/status')
def status():
    return 'I am OK'


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    if y[0] == 0:
        rez = 'low'
    elif y[0] == 1:
        rez = 'medium'
    else:
        rez = 'high'
    return {
        'id': form.id,
        'prediction': rez,
        'price': form.price
    }

