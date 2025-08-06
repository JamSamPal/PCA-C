#include "pca.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
const std::vector<std::vector<double>> PCA::TransposeMatrix(const std::vector<std::vector<double>> &matrix) {
    std::vector<std::vector<double>> transposed_matrix(matrix[0].size(), std::vector<double>(matrix.size(), 0.0));
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[0].size(); ++j) {
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

std::vector<std::vector<double>> PCA::GenerateCovarianceMatrix() {
    // Initialise the transposed matrix
    std::vector<std::vector<double>> centred_data_T = TransposeMatrix(centred_data_);
    std::vector<std::vector<double>> covariance_matrix(num_features_, std::vector<double>(num_features_, 0.0));
    // Calculate covariance matrix
    for (int row = 0; row < num_features_; row++) {
        for (int col = 0; col < num_features_; col++) {
            covariance_matrix[row][col] = 0.0;
            for (int element = 0; element < num_data_points_; element++) {
                covariance_matrix[row][col] += ((centred_data_T[row][element] * centred_data_[element][col]) / (num_data_points_ - 1));
            }
        }
    }

    return covariance_matrix;
}

void PCA::QRDecomposition(std::vector<std::vector<double>> A) {
    // Performs a Q, R decomposition of the covariance matrix
    // Q is an orthogonal matrix and R is an upper-triangular one.
    // Multiplication by an orthogonal matrix leaves the eigenvalues
    // unchanged so the eigenvalues of the product QR are just those
    // or R which, because it is upper-triangular, all lie on the
    // diagonal of R
    // As Q is orthogonal it is equal to its transpose so a lot of the
    // following logic accesses the ROWS of Q

    // This matrix stores the results of gram-schmidt orthogonalisation. Q is in fact nothing
    // but the matrix of orthogonal *unit vectors* found from the gram-schmidt process.
    std::vector<std::vector<double>> u(num_features_, std::vector<double>(num_features_, 0.0));
    u = A;

    // first step of gram-schmidt: u_0 = C[:,0] and thus Q[:,0] = u_0/norm(u_0)
    std::vector<double> norm_u_0 = NormaliseVector(u[0]);
    Q_T_matrix_[0] = norm_u_0;

    for (int i = 1; i < num_features_; i++) {
        for (int j = 0; j < i; j++) {
            std::vector<double> projected_vector = ProjectVector(u[i], Q_T_matrix_[j]);
            // Gram-Schmidt updates each vector and then normalises it.
            for (int k = 0; k < u[0].size(); k++) {
                u[i][k] -= projected_vector[k];
            }
        }
        std::vector<double> norm_u = NormaliseVector(u[i]);
        Q_T_matrix_[i] = norm_u;
    }
    R_matrix_ = MultiplyMatrices(Q_T_matrix_, A);
}

void PCA::QREigenvalues() {
    // Now we iterate the QR decomposition to converge on the eigenvalues
    // encoded in the diagonal of R
    covariance_matrix_ = GenerateCovarianceMatrix();
    std::vector<std::vector<double>> A = TransposeMatrix(covariance_matrix_);

    int i = 0;
    while (i<max_iterations_ & A[0][1]> tolerance_) {
        QRDecomposition(A);
        A = MultiplyMatrices(R_matrix_, TransposeMatrix(Q_T_matrix_));
        i++;
    }
    for (int i = 0; i < eigen_values_.size(); ++i) {
        eigen_values_[i] = A[i][i];
    }

    // Get eigenvalues in descending order
    std::sort(eigen_values_.begin(), eigen_values_.end(), std::greater<>());
}

void PCA::QREigenvectors() {
    // Find the eigenvectors from the approximate eigenvalues
    std::vector<double> random_vector(num_features_);
    random_vector[0] = 1.0;
    for (int i = 0; i < num_features_; i++) {
        // Find the inverse of (A- eigenvalue* Identity) for each eigenvalue
        std::vector<std::vector<double>> A_minus_lambda = SubtractDiagonalMatrix(covariance_matrix_, eigen_values_[i]);
        std::vector<std::vector<double>> inverse = GaussJordanElimination(A_minus_lambda);
        // std::cout << inverse[0][0] << inverse[0][1] << inverse[1][0] << inverse[1][1] << std::endl;
        // Perform the iteration method on a randomly chosen vector, here just a row in Q_T
        eigen_vectors_[i] = IterationMethod(inverse, random_vector);
    }
}

std::vector<double> PCA::IterationMethod(std::vector<std::vector<double>> matrix, std::vector<double> vector) {
    // Iteration method: take (A- eigenvalue* Identity)^-1
    // and multiply by a random vector, b. Repeated applications
    // will eventually yield a b that approximates the eigenvector
    int i = 0;
    int vec_size = vector.size();
    std::vector<double> old_b(vec_size);
    std::vector<double> new_b(vec_size);
    old_b = NormaliseVector(vector);

    while (i < 10000) {
        new_b = MultiplyVector(matrix, old_b);
        old_b = NormaliseVector(new_b);
        i++;
    }

    return old_b;
}

std::vector<double> PCA::MultiplyVector(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector) {
    std::vector<double> output_vector(vector.size());
    for (int row = 0; row < matrix.size(); row++) {
        output_vector[row] = 0.0;
        for (int col = 0; col < vector.size(); col++) {
            output_vector[row] += (matrix[row][col] * vector[col]);
        }
    }

    return output_vector;
}

std::vector<std::vector<double>> PCA::GaussJordanElimination(std::vector<std::vector<double>> matrix) {
    // Gauss-Jordan: going column to column we identify the pivot (the value on the diagonal)
    // and ensure it is non-zero. We then use row addition/subtraction to set all values below
    // the pivot in the same column to zero (forward elimination). We then divide each row by
    // the pivot so that we have 1's along the diagonal. The last step is to perform the backwards
    // elimination to set the remainder of the off-diagonal entries in the upper-half to zero.
    // Performing all these methods but on the identity matrix  instead of our input turns
    // the identity matrix into the inverse we desire
    // Initialise inverse matrix which begins as the identity
    std::vector<std::vector<double>> inverse(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    inverse = SubtractDiagonalMatrix(inverse, -1.0);

    for (int col = 0; col < matrix[0].size(); col++) {
        // Swapping rows if our pivot point is 0
        if (matrix[col][col] == 0) {
            int big = col;
            for (int row = 0; row < matrix.size(); row++) {
                if (abs(matrix[row][col]) > abs(matrix[big][col])) {
                    big = row;
                }
            }
            // Swap rows
            for (int j = 0; j < matrix[0].size(); j++) {
                std::swap(matrix[col][j], matrix[big][j]);
                // Whatever we do to matrix we must also do to inverse
                std::swap(inverse[col][j], inverse[big][j]);
            }
        }
    }

    // Eliminate elements under diagonal (the right-most column has no values under the diagonal so we can skip it)
    for (int col = 0; col < matrix[0].size() - 1; col++) {
        for (int row = col + 1; row < matrix.size(); row++) {
            double scale = matrix[row][col] / matrix[col][col];
            for (int j = 0; j < matrix[0].size(); j++) {
                matrix[row][j] -= scale * matrix[col][j];
                inverse[row][j] -= scale * inverse[col][j];
            }
        }
    }

    // Scale pivots to zero (requires dividing each row by the same value)
    for (int row = 0; row < matrix.size(); row++) {
        double scale = matrix[row][row];
        for (int col = 0; col < matrix[0].size(); col++) {
            matrix[row][col] /= scale;
            inverse[row][col] /= scale;
        }
    }

    // Perform backwards substitution (eliminating values above diagonal)
    // We leverage the fact that when row=col the value is 1. This means that
    // any *value* in a given column, *col*, can be set to zero by row subtraction
    // of the row located at *row*=*col* after that row has been multiplied by *value*
    for (int row = 0; row < matrix.size() - 1; row++) {
        for (int col = row + 1; col < matrix[0].size(); col++) {
            double scale = matrix[row][col];
            for (int k = 0; k < matrix[0].size(); k++) {
                matrix[row][k] -= (matrix[col][k] * scale);
                inverse[row][k] -= (inverse[col][k] * scale);
            }
        }
    }

    return inverse;
}

std::vector<std::vector<double>> PCA::SubtractDiagonalMatrix(std::vector<std::vector<double>> matrix, const double &diag_value) {
    for (int i = 0; i < matrix.size(); i++) {
        matrix[i][i] -= diag_value;
    }
    return matrix;
}

double PCA::CalculateNorm(const std::vector<double> &u, const std::vector<double> &v) {
    double norm = 0;
    for (int val = 0; val < u.size(); val++) {
        norm += u[val] * v[val];
    }

    return norm;
}

std::vector<double> PCA::NormaliseVector(std::vector<double> u) {
    double norm = std::sqrt(CalculateNorm(u, u));
    for (int val = 0; val < u.size(); val++) {
        u[val] /= norm;
    }

    return u;
}

std::vector<double> PCA::ProjectVector(const std::vector<double> &u, const std::vector<double> &e) {
    double norm_1 = CalculateNorm(u, e);
    double norm_2 = CalculateNorm(e, e);
    std::vector<double> proj_vector(e.size());
    for (int val = 0; val < e.size(); val++) {
        proj_vector[val] = e[val] * (norm_1 / norm_2);
    }
    return proj_vector;
}

std::vector<std::vector<double>> PCA::MultiplyMatrices(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b) {
    std::vector<std::vector<double>> output_matrix(a.size(), std::vector<double>(b[0].size(), 0.0));
    for (int row = 0; row < a.size(); row++) {
        for (int col = 0; col < b[0].size(); col++) {
            output_matrix[row][col] = 0.0;
            for (int element = 0; element < a[0].size(); element++) {
                output_matrix[row][col] += (a[row][element] * b[element][col]);
            }
        }
    }

    return output_matrix;
}

void PCA::SaveToFile(const std::string &filename_prefix) {
    std::ofstream eigval_file(filename_prefix + "_eigenvalues.txt");
    for (int i = 0; i < eigen_values_.size(); i++) {
        eigval_file << eigen_values_[i] << "\n";
    }
    std::ofstream eigvec_file(filename_prefix + "_eigenvectors.txt");
    for (int i = 0; i < eigen_vectors_.size(); i++) {
        for (int j = 0; j < eigen_vectors_[0].size(); j++) {
            eigvec_file << eigen_vectors_[i][j] << " ";
        }
        eigvec_file << "\n";
    }
}