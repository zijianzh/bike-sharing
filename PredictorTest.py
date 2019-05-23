import unittest
import pandas as pd
from prediction import Predictor
from sklearn.base import BaseEstimator

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.features = pd.read_csv('test_dataset.csv', sep=',')
        self.preditor = Predictor(self.features)

    def test_load_model(self):
        self.assertIsInstance(self.preditor._load_model('lm.joblib'), BaseEstimator)

    def test_predict(self):
        predictions = self.preditor.predict()
        self.assertIsInstance(predictions, list)
        self.assertIs(len(predictions), self.features.shape[0])
        self.assertGreaterEqual(min(predictions), 0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
