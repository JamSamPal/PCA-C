import numpy as np
import subprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

executable = Path(__file__).parent.parent / "c++/pca_calc"
subprocess.run(f"{str(executable)}", shell=True)

# Read eigenvalues
eigenvalues_test = np.loadtxt("test_eigenvalues.txt")
eigenvalues_test.sort()
# Read eigenvectors
eigenvectors_test = np.loadtxt("test_eigenvectors.txt")


def test_simple():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    assert eigenvalues.all() == eigenvalues_test.all()
