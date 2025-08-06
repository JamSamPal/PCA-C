#ifndef C52DD91C_DDBE_4C77_9498_12FE881D8350
#define C52DD91C_DDBE_4C77_9498_12FE881D8350

#include <string>
#include <vector>
class PCA {
public:
    PCA(const int &num_features, const int &num_data_points, const std::vector<std::vector<double>> &input_data) : num_features_(num_features), num_data_points_(num_data_points), matrix_(input_data), matrix_T_(num_features, std::vector<double>(num_data_points, 0.0)), centred_data_(num_data_points, std::vector<double>(num_features, 0.0)), centred_data_T_(num_features, std::vector<double>(num_data_points, 0.0)), R_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), Q_T_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), Q_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), covariance_matrix_(num_features_, std::vector<double>(num_features_, 0.0)), covariance_matrix_T_(num_features_, std::vector<double>(num_features_, 0.0)), eigen_values_(num_features), eigen_vectors_(num_features_, std::vector<double>(num_features_, 0.0)) {
        TransposeMatrix(matrix_, matrix_T_);
        CentreData();
        TransposeMatrix(centred_data_, centred_data_T_);
        GenerateCovarianceMatrix();
        TransposeMatrix(covariance_matrix_, covariance_matrix_T_);
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
    // a common design pattern will be to use buffers to avoid reallocation
    void TransposeMatrix(const std::vector<std::vector<double>> &matrix, std::vector<std::vector<double>> &transposed);
    void NormaliseVector(std::vector<double> &u);
    void ProjectVector(const std::vector<double> &u, const std::vector<double> &e, std::vector<double> &projected_vector);
    void MultiplyMatrices(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, std::vector<std::vector<double>> &c);
    void MultiplyVector(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector, std::vector<double> &output_vector);
    void GaussJordanElimination(std::vector<std::vector<double>> &input_matrix, std::vector<std::vector<double>> &inverse);
    std::vector<double> IterationMethod(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector);
    void SubtractDiagonalMatrix(const std::vector<std::vector<double>> &matrix, const double &diag_value, std::vector<std::vector<double>> &output);
    // pca
    std::vector<std::vector<double>> matrix_;
    std::vector<std::vector<double>> matrix_T_;
    void CentreData();
    std::vector<std::vector<double>> centred_data_;
    std::vector<std::vector<double>> centred_data_T_;
    void GenerateCovarianceMatrix();
    std::vector<std::vector<double>> covariance_matrix_;
    std::vector<std::vector<double>> covariance_matrix_T_;
    std::vector<double> eigen_values_;
    std::vector<std::vector<double>> eigen_vectors_;
    std::vector<std::vector<double>> Q_T_matrix_;
    std::vector<std::vector<double>> Q_matrix_;
    std::vector<std::vector<double>> R_matrix_;
    void QRDecomposition(const std::vector<std::vector<double>> &A);
};

#endif /* C52DD91C_DDBE_4C77_9498_12FE881D8350 */
