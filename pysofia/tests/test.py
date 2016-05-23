import numpy as np
from scipy import stats
from pysofia import svm_train, svm_predict, learner_type, loop_type, eta_type

def test_1():
    np.random.seed(0)
    X = np.random.randn(200, 5)
    query_id = np.ones(len(X))
    w = np.random.randn(5)
    y = np.dot(X, w)
    coef = svm_train(X, y, query_id, 1., X.shape[0], X.shape[1], learner_type.pegasos,
     loop_type.rank, eta_type.basic_eta, max_iter=100)
    prediction = svm_predict(X, coef)
    tau, _ = stats.kendalltau(y, prediction)
    assert np.abs(1 - tau) > 1e-3

if __name__ == '__main__':
    test_1()
