from sklearn import preprocessing
from joblib import dump, load
import pandas as pd

enc = load('enc.joblib')
print(enc)

lm = load('lm.joblib')
print(lm)

svr = load('svr.joblib')
print(svr)

xgb = load('xgb.joblib')
print(xgb)

test_input = pd.read_csv('test_dataset.csv', sep=',')
print(test_input)
test_features = test_input.drop(['instant','dteday'], axis = 1)
print(test_features)

test_features.yr = test_features.yr/100
test_features.mnth = test_features.mnth/12
test_features.hr = test_features.hr/24
test_features.weekday = test_features.weekday/7



ohe_test_features = enc.transform(test_features)
print('### ohe test features #####')
print(ohe_test_features)
predictions = xgb.predict(ohe_test_features)
print(predictions)