#ifndef C52DD91C_DDBE_4C77_9498_12FE881D8350
#define C52DD91C_DDBE_4C77_9498_12FE881D8350

#include <string>
#include <vector>
class PCA {
public:
    PCA(const int &num_features, const int &num_data_points, const std::vector<std::vector<double>> &input_data) : num_features_(num_features), num_data_points_(num_data_points), matrix_(input_data), centred_data_(num_data_points, std::vector<double>(num_features, 0.0)), R_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), Q_T_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), covariance_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), eigen_values_(num_features), eigen_vectors_(num_features_, std::vector<double>(num_features_, 0.0)) {
        CentreData();
    };

    void QREigenvalues();
    void QREigenvectors();
    void SaveToFile(const std::string &filename_prefix);

private:
    int num_features_;
    int num_data_points_;
    const double tolerance_ = 0.00001;
    const int max_decompositions_ = 10000;
    const int max_iterations_ = 10000;
    // general matrix manipulation methods
    double CalculateNorm(const std::vector<double> &u, const std::vector<double> &v);
    const std::vector<std::vector<double>> TransposeMatrix(const std::vector<std::vector<double>> &matrix);
    std::vector<double> NormaliseVector(std::vector<double> u);
    std::vector<double> ProjectVector(const std::vector<double> &u, const std::vector<double> &e);
    std::vector<std::vector<double>> MultiplyMatrices(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b);
    std::vector<double> MultiplyVector(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector);
    std::vector<std::vector<double>> GaussJordanElimination(std::vector<std::vector<double>> matrix);
    std::vector<double> IterationMethod(std::vector<std::vector<double>> matrix, std::vector<double> vector);
    std::vector<std::vector<double>> SubtractDiagonalMatrix(std::vector<std::vector<double>> matrix, const double &diag_value);
    // pca
    std::vector<std::vector<double>> matrix_;
    void CentreData();
    std::vector<std::vector<double>> centred_data_;
    std::vector<std::vector<double>> GenerateCovarianceMatrix();
    std::vector<std::vector<double>> covariance_matrix_;
    std::vector<double> eigen_values_;
    std::vector<std::vector<double>> eigen_vectors_;
    std::vector<std::vector<double>> Q_T_matrix_;
    std::vector<std::vector<double>> R_matrix_;
    void QRDecomposition(std::vector<std::vector<double>> A);
};

#endif /* C52DD91C_DDBE_4C77_9498_12FE881D8350 */
