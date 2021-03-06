import unittest
import pandas as pd
from prediction import Predictor
from sklearn.base import BaseEstimator

class PredictorTest(unittest.TestCase):

    def setUp(self):
        self.features = pd.read_csv('test_dataset.csv', sep=',')
        self.preditor = Predictor(self.features)

    def test_load_model(self):
        self.assertIsInstance(self.preditor._load_model('lm.joblib'), BaseEstimator)

    def test_predict(self):
        predictions = self.preditor.predict()
        self.assertIsInstance(predictions, list)
        # test the prediction results has same instances as input instances
        self.assertIs(len(predictions), self.features.shape[0])
        # test the predictions all greater than zero
        self.assertGreaterEqual(min(predictions), 0)



if __name__ == '__main__':
    unittest.main()
