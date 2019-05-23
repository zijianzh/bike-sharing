from prediction import Predictor
import pandas as pd

#df = pd.read_csv('test_dataset.csv', sep=',')
#print(type(df))
test_predictor = Predictor(pd.read_csv('test_dataset.csv', sep=','))
test_prediction = test_predictor.predict()
print(test_prediction)