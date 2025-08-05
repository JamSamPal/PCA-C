#include "pca.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
const std::vector<std::vector<double>> PCA::TransposeMatrix(const std::vector<std::vector<double>> &matrix) {
    std::vector<std::vector<double>> transposed_matrix(num_features_, std::vector<double>(num_data_points_, 0.0));
    for (int i = 0; i < num_data_points_; ++i) {
        for (int j = 0; j < num_features_; ++j) {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }
    return transposed_matrix;
}

void PCA::CentreData() {
    // We loop through the transposed matrix where each row is a feature
    // Calculating the mean and sd of this feature we can then centre and
    // normalise the data of the original
    std::vector<std::vector<double>> matrix_T = TransposeMatrix(matrix_);
    for (int row = 0; row < num_features_; row++) {

        double sum = std::accumulate(std::begin(matrix_T[row]), std::end(matrix_T[row]), 0.0);
        double m = sum / matrix_T[row].size();

        double var = 0;
        for (int col = 0; col < num_data_points_; col++) {
            var += (matrix_T[row][col] - m) * (matrix_T[row][col] - m);
        }
        var /= num_data_points_;
        double sd = std::sqrt(var);

        // In the original, untransposed matrix "row" now indexes the columns
        for (int row_T = 0; row_T < num_data_points_; row_T++) {
            centred_data_[row_T][row] = (matrix_[row_T][row] - m) / sd;
        }
    }
}

void PCA::GenerateCovarianceMatrix() {
    // Initialise the transposed matrix
    std::vector<std::vector<double>> centred_data_T = TransposeMatrix(centred_data_);

    // Calculate covariance matrix
    for (int row = 0; row < num_features_; row++) {
        for (int col = 0; col < num_features_; col++) {
            covariance_matrix_(row, col) = 0.0;
            for (int element = 0; element < num_data_points_; element++) {
                covariance_matrix_(row, col) += ((centred_data_T[row][element] * centred_data_[element][col]) / (num_data_points_ - 1));
            }
        }
    }
}

void PCA::ComputeEigenvaluesEigenvectors() {
    // Covariance matrix will be self adjoint
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver;
    eigensolver.compute(covariance_matrix_);
    eigen_values_ = eigensolver.eigenvalues().real();
    eigen_vectors_ = eigensolver.eigenvectors().real();
}

void PCA::SaveToFile(const std::string &filename_prefix) {
    std::ofstream eigval_file(filename_prefix + "_eigenvalues.txt");
    for (int i = 0; i < eigen_values_.size(); ++i) {
        eigval_file << eigen_values_[i] << "\n";
    }

    std::ofstream eigvec_file(filename_prefix + "_eigenvectors.txt");
    eigvec_file << eigen_vectors_ << "\n";
}