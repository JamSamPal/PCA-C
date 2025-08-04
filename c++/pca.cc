#include "pca.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
const std::vector<std::vector<double>> PCA::_Transpose_matrix(const std::vector<std::vector<double>> &matrix) {
    std::vector<std::vector<double>> transposed_matrix(num_features, std::vector<double>(num_data_points, 0.0));
    for (int i = 0; i < num_data_points; ++i) {
        for (int j = 0; j < num_features; ++j) {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }
    return transposed_matrix;
}

void PCA::_Centre_Data() {
    // We loop through the transposed matrix where each row is a feature
    // Calculating the mean and sd of this feature we can then centre and
    // normalise the data of the original matrix
    for (int row = 0; row < num_features; row++) {

        double sum = std::accumulate(std::begin(matrix_T[row]), std::end(matrix_T[row]), 0.0);
        double m = sum / matrix_T[row].size();

        double var = 0;
        for (int col = 0; col < num_data_points; col++) {
            var += (matrix_T[row][col] - m) * (matrix_T[row][col] - m);
        }
        var /= num_data_points;
        double sd = std::sqrt(var);

        // In the original, untransposed matrix "row" now indexes the columns
        for (int row_T = 0; row_T < num_data_points; row_T++) {
            centred_data[row_T][row] = (matrix[row_T][row] - m) / sd;
        }
    }
    // Initialise the transposed matrix
    centred_data_T = _Transpose_matrix(centred_data);
}

void PCA::_Generate_Covariance_Matrix() {
    for (int row = 0; row < num_features; row++) {
        for (int col = 0; col < num_features; col++) {
            covariance_matrix(row, col) = 0.0;
            for (int element = 0; element < num_data_points; element++) {
                covariance_matrix(row, col) += ((centred_data_T[row][element] * centred_data[element][col]) / (num_data_points - 1));
            }
        }
    }
}

void PCA::Compute_Eigenvalues_Eigenvectors() {
    // Covariance matrix will be self adjoint
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver;
    eigensolver.compute(covariance_matrix);
    eigen_values = eigensolver.eigenvalues().real();
    eigen_vectors = eigensolver.eigenvectors().real();
}

void PCA::Save_to_file(const std::string &filename_prefix) {
    std::ofstream eigval_file(filename_prefix + "_eigenvalues.txt");
    for (int i = 0; i < eigen_values.size(); ++i) {
        eigval_file << eigen_values[i] << "\n";
    }

    std::ofstream eigvec_file(filename_prefix + "_eigenvectors.txt");
    eigvec_file << eigen_vectors << "\n";
}