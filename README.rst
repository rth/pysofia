PySofia
=======

.. image:: https://travis-ci.org/rth/pysofia.svg?branch=master
    :target: https://travis-ci.org/rth/pysofia
.. image:: https://ci.appveyor.com/api/projects/status/jhn2d8n8y836pnj8/branch/master?svg=true
    :target: https://ci.appveyor.com/project/rth/pysofia/branch/master

PySofia is a python wrapper around the methods present in the C++ sofia-ml library. These include Stochastic Gradient Descent implementations of some ranking algorithms, notably RankSVM.

Dependencies
------------

- cython >= 0.17 (previous versions will not work)
- numpy
- sklearn >= 0.15
- six
- enum34 (for Python versions before 3.4)
- A C++ compiler (gcc will do)

You will not need to install sofia-ml since it is incuded in this distribution

Installation
============

::

    $ pip install -U git+https://github.com/rth/pysofia.git


Methods
=======

pysofia.train_svm

    Trains a model using stochastic gradient descent. See docstring for
    more details.

pysofia.compat.RankSVM implements an estimator following the conventions
used in scikit-learn.

Development
===========

Check out the latest version at github: https://github.com/rth/pysofia/

License
=======

Apache License 2.0

Authors
=======

PySofia is the work of Fabian Pedregosa, currently maintained by Roman Yurchak. The sofia-ml library is written by D. Sculley.
