import os.path
import dill

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
print(os.getcwd())
with open('model/best_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str | None
    device_category: str
    device_os: str | None
    device_brand: str | None
    device_model: str | None
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    result: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'result': y[0]
    }
