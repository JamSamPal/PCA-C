# PCA-C
Principal Component Analysis done in C++. An exercise in learning about high performance computing.

- Close attention is payed to avoiding dynamic allocations, ensuring optimal memory usage.
- However, libraries like "Eigen" are avoided to put emphasis on understanding the mathematics.

## Background
Given points in an N-dimensional vector space, what directions in this space capture the variance of the points the best?
Specifically could we find a set of D orthogonal unit vectors in this space which, when the original points are projected onto them,
best retain the variance in the data?

These questions are answered by PCA.

We take the say M points to from an M by N dimensional matrix, X.
Projecting this matrix onto an arbitrary vector, v, is done by taking X.v
The variance of this projection can be found by taking ((X.v).T).(X.v) which results in (v.T).C.v
where C = (X.T).(X) is the covariance matrix.

We wish to maximise ((X.v).T).(X.v), and so maximise the variance on this projected vector v subject to that vector
being a unit vector: (v.T).v -1 = 0. We can formulate this as a classic lagrange multiplier problem, asking us to maximise

L(v) = (v.T).C.v - \lambda ((v.T).v -1)

The derivative w.r.t to v maximises this function and we find we must solve

C.v = \lambda v

which is an eigenvalue problem. The eigenvalues represent the variance and so the largest D eigenvalues and their corresponding eigenvectors represent the unit vectors that best retain the variance.


## To run
./pca_calc (After building executable)

pip install -e .

pytest (runs executable and then runs tests)
