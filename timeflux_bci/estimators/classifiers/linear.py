from sklearn.linear_model import RidgeClassifier
from scipy.special import softmax

decision_function = getattr(RidgeClassifier, 'decision_function')
def predict_proba(X, y): return softmax(RidgeClassifier.decision_function(X, y))
setattr(RidgeClassifier, 'predict_proba', predict_proba)
