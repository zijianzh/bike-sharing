
from joblib import dump, load
lm = load('lm.joblib')
print(lm)

svr = load('svr.joblib')
print(svr)