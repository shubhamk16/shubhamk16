import joblib
import traceback
import pandas as pd
import numpy as np
import logging

# 1. Library imports
import uvicorn
from fastapi import FastAPI

# 2. Create the app object
app = FastAPI()

lr = joblib.load("app/model.pkl") # Load "model.pkl"
logging.info('Model loaded')
model_columns = joblib.load("app/model_columns.pkl") # Load "model_columns.pkl"
logging.info('Model columns loaded')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, Glad to see your Interest In Loan prediction model'}

@app.get('/sample example')
def sample_example():
    """ get call example 

    Returns:
        list of dictionary: sample expample of the data
    """
    return [
             {"Loan_ID": "LP001003",
             "Gender": "Male",
             "Married": "Yes",
             "Dependents": "1",
             "Education": "Graduate",
             "Self_Employed": "No",
             "ApplicantIncome": 4583,
             "CoapplicantIncome": 1508.0,
             "LoanAmount": 128.0,
             "Loan_Amount_Term": 360.0,
             "Credit_History": 1.0,
             "Property_Area": "Rural"}
            ]


@app.post('/preprocess')
def preprocess(request:list):
    """ preprocess the data"""
    try:
        query = pd.get_dummies(pd.DataFrame(request))
        query = query.reindex(columns=model_columns, fill_value=0)
        preprocessed_query = (query.to_dict(orient="rows"))
        return ({'preprocessed data': preprocessed_query })

    except:

        return ({'trace': traceback.format_exc()})
    
    
@app.post('/predict')
def predict(request:list):
    """ predict the data"""
    if lr:
        try:
            preprocessed_query = preprocess(request)
            input = pd.DataFrame(preprocessed_query['preprocessed data'])
            prediction = list(lr.predict(input))

            return ({'prediction': str(prediction)})

        except:

            return ({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

    
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)
    
    
    