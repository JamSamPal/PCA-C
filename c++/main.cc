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

    pca.QREigenvalues();
    pca.QREigenvectors();
    std::string prefix = "test";
    pca.SaveToFile(prefix);
    return 0;
}