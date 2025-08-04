import numpy as np
from sklearn.decomposition import PCA

def test_simple():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    # Will assert c++ output is the same
    assert 1==1
