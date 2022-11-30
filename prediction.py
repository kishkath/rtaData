import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def get_predictions(data,model):
    '''
    Returns the predictions of model
    :param data: .csv/values
    :param model: .joblib
    :return: Slight Injury/Fatal Injury/Serious Injury
    '''
    return model.predict(data)