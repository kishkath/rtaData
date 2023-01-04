import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.read_csv("Dataset/cleanedData.csv")

def encoding(input_val,colName):
    le = LabelEncoder()
    values = data[colName].unique()
    data[colName] = le.fit_transform(data[colName])
    dicts = dict(zip(values, data[colName].unique()))
    return dicts[input_val]

def get_predictions(data,model):
    '''
    Returns the predictions of model
    :param data: .csv/values
    :param model: .joblib
    :return: Slight Injury/Fatal Injury/Serious Injury
    '''
    return model.predict(data)
