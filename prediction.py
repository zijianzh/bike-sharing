import pandas as pd
import numpy as np
from joblib import load

class Predictor:


    def __init__(self, predict_X):
        self.estimitor = self.__load_model('xgb.joblib')
        self.enc = self.__load_model('enc.joblib')
        self.predict_X = predict_X

    def __load_model(self, filename =''):
        return load(filename)

    def predict(self):
        raw_predictions = self.estimitor.predict(self.__fit_features())
        print(raw_predictions)
        # take prediction results that smaller then zero as zero
        # make float prediction results to int
        predictions = [np.rint(prediction).astype(int) if prediction >=0 else 0 for prediction in raw_predictions ]
        return predictions

    def __fit_features(self):
        scaled_X = self.__scale_num_features(self.predict_X.drop(['instant','dteday'], axis = 1))
        return self.enc.transform(scaled_X)

    def __scale_num_features(self, features):
        print(features)
        features.yr = features.yr / 100
        features.mnth = features.mnth / 12
        features.hr = features.hr / 24
        features.weekday = features.weekday / 7
        return features

