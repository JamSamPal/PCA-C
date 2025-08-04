#include "pca.hpp"

int main() {
    std::vector<std::vector<double>> X = {
        {-1, -1},
        {-2, -1},
        {-3, -2},
        {1, 1},
        {2, 1},
        {3, 2}};
    PCA pca(2, 6, X);

    pca.Compute_Eigenvalues_Eigenvectors();
    std::string prefix = "test";
    pca.Save_to_file(prefix);
    return 0;
}