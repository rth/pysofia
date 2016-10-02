from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys, tempfile
import six
import numpy as np
from sklearn import datasets


from enum import Enum


class learner_type(Enum):
    r"""
    define learner type
    """
    pegasos = 0
    margin_perceptron = 1
    passive_aggressive = 2
    logreg_pegasos = 3
    logreg = 4
    lms_regression = 5
    sgd_svm = 6
    romma = 7

class loop_type(Enum):
    r"""
    define loop type
    """
    rank = 'rank'
    roc = 'roc'
    combined_ranking = 'combined-ranking'
    stochastic = 'stochastic'
    balanced_stochastic = 'balanced-stochastic'

class eta_type(Enum):
    r"""
    define eta type
    """
    basic_eta = 0
    pegasos_eta = 1
    constant = 2

class predict_type(Enum):
    r"""
    define predict type
    """
    linear = 'linear'
    logistic = 'logistic'

def svm_train(X, y, b, alpha, n_samples, n_features, learner, loop, eta,
              max_iter=100, step_probability=0.5):
    """
    Minimizes an expression of the form

        Loss(X, y, b) + 0.5 * alpha * (||w|| ** 2)

    where Loss is an Hinge loss defined on pairs of images

    Parameters
    ----------
    X : input data

    y : target labels

    b : blocks (aka query_id)

    alpha: float

    loop : {'rank', 'combined-ranking', 'roc', 'stochastic', 'balanced-stochastic'}

    Returns
    -------
    coef

    """
    from pysofia import _sofia_ml
    if isinstance(X, six.string_types):
        if n_features is None:
            n_features = 2**17 # the default in sofia-ml TODO: parse file to see
        w = _sofia_ml.train(X, n_features, alpha, max_iter, False,
                            learner.value, loop.value, eta.value, step_probability)
    elif isinstance(X, np.ndarray):
        if n_features is None:
            n_features = X.shape[1]

        if n_samples is None:
            n_samples = X.shape[0]

        w = _sofia_ml.train_fast(np.float64(X), np.float64(y), n_samples, n_features, alpha, max_iter, False,
                                 learner.value, loop.value, eta.value, step_probability)
    else:
        if n_features is None:
            n_features = X.shape[1]

        with tempfile.NamedTemporaryFile() as f:
            datasets.dump_svmlight_file(X, y, f.name, query_id=b)
            w = _sofia_ml.train(f.name, n_features, alpha, max_iter, False,
                                learner.value, loop.value, eta.value, step_probability)
    return w

def svm_predict(data, coef, predict_type=predict_type.linear, blocks=None):
    # TODO: isn't query_id in data ???
    from pysofia import _sofia_ml
    s_coef = ' '.join(['%.5f' % e for e in coef[:-1]])

    if six.PY3:
        s_coef = s_coef.encode('utf-8')

    if isinstance(data, six.string_types):
        return _sofia_ml.predict(data, s_coef, predict_type.value, False)
    elif isinstance(data, np.ndarray):
        y = np.ones(data.shape[0])
        return _sofia_ml.predict_fast(np.float64(data), y, data.shape[0], data.shape[1],
                                      s_coef, predict_type.value, False)
    else:
        X = np.asarray(data)
        if blocks is None:
            blocks = np.ones(X.shape[0])

        with tempfile.NamedTemporaryFile() as f:
            y = np.ones(X.shape[0])
            datasets.dump_svmlight_file(X, y, f.name, query_id=blocks)
            prediction = _sofia_ml.predict(f.name, s_coef, predict_type.value, False)
        return prediction

def svm_update(X, y, coef, alpha, n_samples, n_features, learner, loop, eta,
              max_iter=100, step_probability=0.5):
    """
    Update SVM using new data

    Parameters
    ----------
    X : input data

    y : target labels

    alpha: float

    loop : {'rank', 'combined-ranking', 'roc', 'stochastic', 'balanced-stochastic'}

    Returns
    -------
    coef

    """
    from pysofia import _sofia_ml
    if isinstance(X, np.ndarray):
        if n_features is None:
            n_features = X.shape[1]

        if n_samples is None:
            n_samples = X.shape[0]

        w = _sofia_ml.update_fast(np.float64(X), np.float64(y), np.float64(coef), n_samples, n_features, alpha,
                                  max_iter, False, learner.value, loop.value, eta.value, step_probability)
    else:
        raise ValueError("Input should be matrix!")
    return w
