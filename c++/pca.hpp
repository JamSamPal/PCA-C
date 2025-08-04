#ifndef C52DD91C_DDBE_4C77_9498_12FE881D8350
#define C52DD91C_DDBE_4C77_9498_12FE881D8350

#include <Eigen/Dense>
#include <vector>
class PCA {
public:
    PCA(const int &num_features, const int &num_data_points, std::vector<std::vector<double>> &input_data) : num_features(num_features), num_data_points(num_data_points), matrix(num_data_points, std::vector<double>(num_features, 0.0)), matrix_T(num_features, std::vector<double>(num_data_points, 0.0)), centred_data(num_data_points, std::vector<double>(num_features, 0.0)), centred_data_T(num_features, std::vector<double>(num_data_points, 0.0)), covariance_matrix(num_features, num_features) {
        matrix = input_data;
        matrix_T = _Transpose_matrix(matrix);
        _Centre_Data();
        _Generate_Covariance_Matrix();
    };
    void Compute_Eigenvalues_Eigenvectors();
    void Save_to_file(const std::string &filename_prefix);

private:
    int num_features;
    int num_data_points;
    std::vector<std::vector<double>> matrix;
    std::vector<std::vector<double>> matrix_T;
    const std::vector<std::vector<double>> _Transpose_matrix(const std::vector<std::vector<double>> &matrix);
    void _Centre_Data();
    std::vector<std::vector<double>> centred_data;
    std::vector<std::vector<double>> centred_data_T;
    void _Generate_Covariance_Matrix();
    Eigen::MatrixXd covariance_matrix;
    Eigen::VectorXd eigen_values;
    Eigen::MatrixXd eigen_vectors;
};

#endif /* C52DD91C_DDBE_4C77_9498_12FE881D8350 */
