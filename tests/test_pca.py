import numpy as np
import subprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

executable = Path(__file__).parent.parent / "c++/pca_calc"
subprocess.run(f"{str(executable)}", shell=True)

# Read eigenvalues
eigenvalues_test = np.loadtxt("test_eigenvalues.txt")
# # Read eigenvectors
eigenvectors_test = np.loadtxt("test_eigenvectors.txt")


def test_simple():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    # test that implementation gets within a good degree of tolerance
    for i in range(len(X[0])):
        assert abs(eigenvalues_test[i] - eigenvalues[i]) < 0.00001

    for i in range(len(X[0])):
        eigenvector_plus = eigenvectors[i]
        eigenvector_minus = -eigenvectors[i]
        for j in range(len(X[0])):
            assert (
                eigenvector_plus[j] - eigenvectors_test[i][j] < 0.001
                or eigenvector_minus[j] - eigenvectors_test[i][j] < 0.001
            )
