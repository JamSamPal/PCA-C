#ifndef C52DD91C_DDBE_4C77_9498_12FE881D8350
#define C52DD91C_DDBE_4C77_9498_12FE881D8350

#include <Eigen/Dense>
#include <vector>
class PCA {
public:
    PCA(const int &num_features, const int &num_data_points, const std::vector<std::vector<double>> &input_data) : num_features_(num_features), num_data_points_(num_data_points), matrix_(input_data), centred_data_(num_data_points, std::vector<double>(num_features, 0.0)), covariance_matrix_(num_features, num_features) {
        CentreData();
        GenerateCovarianceMatrix();
    };
    void ComputeEigenvaluesEigenvectors();
    void SaveToFile(const std::string &filename_prefix);

private:
    int num_features_;
    int num_data_points_;
    std::vector<std::vector<double>> matrix_;
    const std::vector<std::vector<double>> TransposeMatrix(const std::vector<std::vector<double>> &matrix);
    void CentreData();
    std::vector<std::vector<double>> centred_data_;
    void GenerateCovarianceMatrix();
    Eigen::MatrixXd covariance_matrix_;
    Eigen::VectorXd eigen_values_;
    Eigen::MatrixXd eigen_vectors_;
};

#endif /* C52DD91C_DDBE_4C77_9498_12FE881D8350 */
