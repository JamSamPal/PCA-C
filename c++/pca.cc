#include "pca.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

void PCA::CentreData() {
    // We loop through the transposed matrix where each row is a feature
    // Calculating the mean and sd of this feature we can then centre and
    // normalise the data of the original
    for (int row = 0; row < num_features_; row++) {

        double sum = 0.0, sum_sq = 0.0;
        for (double x : matrix_T_[row]) {
            sum += x;
            sum_sq += x * x;
        }
        double m = sum / num_data_points_;
        double var = sum_sq / num_data_points_ - m * m;
        double sd = std::sqrt(var);

        // In the original, untransposed matrix "row" now indexes the columns
        for (int row_T = 0; row_T < num_data_points_; row_T++) {
            centred_data_[row_T][row] = (matrix_[row_T][row] - m) / sd;
        }
    }
}

void PCA::GenerateCovarianceMatrix() {
    // Calculate covariance matrix (symmetric so we can half the work needed)
    for (int row = 0; row < num_features_; row++) {
        for (int col = row; col < num_features_; col++) {
            double value = 0.0;
            for (int element = 0; element < num_data_points_; element++) {
                value += ((centred_data_T_[row][element] * centred_data_[element][col]) / (num_data_points_ - 1));
            }
            covariance_matrix_[row][col] = value;
            covariance_matrix_[col][row] = value;
        }
    }
}

void PCA::QRDecomposition(const std::vector<std::vector<double>> &A) {
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
    NormaliseVector(u[0]);
    Q_T_matrix_[0] = u[0];
    std::vector<double> projected_vector_buffer(num_features_);

    for (int i = 1; i < num_features_; i++) {
        for (int j = 0; j < i; j++) {
            ProjectVector(u[i], Q_T_matrix_[j], projected_vector_buffer);
            // Gram-Schmidt updates each vector and then normalises it.
            for (int k = 0; k < u[0].size(); k++) {
                u[i][k] -= projected_vector_buffer[k];
            }
        }
        NormaliseVector(u[i]);
        Q_T_matrix_[i] = u[i];
    }
    MultiplyMatrices(Q_T_matrix_, A, R_matrix_);
}

void PCA::QREigenvalues() {
    // Now we iterate the QR decomposition to converge on the eigenvalues
    // encoded in the diagonal of R
    int i = 0;
    while (i<max_iterations_ & covariance_matrix_T_[0][1]> tolerance_) {
        QRDecomposition(covariance_matrix_T_);
        TransposeMatrix(Q_T_matrix_, Q_matrix_);
        MultiplyMatrices(R_matrix_, Q_matrix_, covariance_matrix_T_);
        i++;
    }
    for (int i = 0; i < eigen_values_.size(); ++i) {
        eigen_values_[i] = covariance_matrix_T_[i][i];
    }

    // Get eigenvalues in descending order
    std::sort(eigen_values_.begin(), eigen_values_.end(), std::greater<>());
}

void PCA::QREigenvectors() {
    // Find the eigenvectors from the approximate eigenvalues
    std::vector<double> random_vector(num_features_);
    std::vector<std::vector<double>> A_minus_lambda(num_features_, std::vector<double>(num_features_, 0.0));
    std::vector<std::vector<double>> inverse(num_features_, std::vector<double>(num_features_, 0.0));
    random_vector[0] = 1.0;
    for (int i = 0; i < num_features_; i++) {
        // Find the inverse of (A- eigenvalue* Identity) for each eigenvalue
        SubtractDiagonalMatrix(covariance_matrix_, eigen_values_[i], A_minus_lambda);
        GaussJordanElimination(A_minus_lambda, inverse);
        // Perform the iteration method on a randomly chosen vector
        eigen_vectors_[i] = IterationMethod(inverse, random_vector);
    }
}

std::vector<double> PCA::IterationMethod(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector) {
    // Iteration method: take (A- eigenvalue* Identity)^-1
    // and multiply by a random vector, b. Repeated applications
    // will eventually yield a b that approximates the eigenvector
    int i = 0;
    int vec_size = vector.size();
    std::vector<double> old_b(vec_size);
    std::vector<double> new_b(vec_size);
    old_b = vector;
    NormaliseVector(old_b);

    while (i < 10000) {
        MultiplyVector(matrix, old_b, new_b);
        NormaliseVector(new_b);
        old_b.swap(new_b);
        i++;
    }

    return old_b;
}

void PCA::TransposeMatrix(const std::vector<std::vector<double>> &matrix, std::vector<std::vector<double>> &transposed) {
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = 0; j < matrix[0].size(); ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
}

void PCA::MultiplyVector(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector, std::vector<double> &output_vector) {
    for (int row = 0; row < matrix.size(); row++) {
        output_vector[row] = 0.0;
        for (int col = 0; col < vector.size(); col++) {
            output_vector[row] += (matrix[row][col] * vector[col]);
        }
    }
}

void PCA::GaussJordanElimination(std::vector<std::vector<double>> &matrix, std::vector<std::vector<double>> &inverse) {
    // Gauss-Jordan: going column to column we identify the pivot (the value on the diagonal)
    // and ensure it is non-zero. We then use row addition/subtraction to set all values below
    // the pivot in the same column to zero (forward elimination). We then divide each row by
    // the pivot so that we have 1's along the diagonal. The last step is to perform the backwards
    // elimination to set the remainder of the off-diagonal entries in the upper-half to zero.
    // Performing all these methods but on the identity matrix  instead of our input turns
    // the identity matrix into the inverse we desire
    // Initialise inverse matrix which begins as the identity
    // Initialize inverse to Identity
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix.size(); j++) {
            inverse[i][j] = (i == j ? 1.0 : 0.0);
        }
    }

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
}

void PCA::SubtractDiagonalMatrix(const std::vector<std::vector<double>> &matrix, const double &diag_value, std::vector<std::vector<double>> &output) {
    for (int i = 0; i < matrix.size(); i++) {
        output[i][i] = covariance_matrix_[i][i] - diag_value;
        for (int j = 0; j < matrix.size(); j++) {
            if (i != j)
                output[i][j] = covariance_matrix_[i][j];
        }
    }
}

double PCA::CalculateNorm(const std::vector<double> &u, const std::vector<double> &v) {
    double norm = 0;
    for (int val = 0; val < u.size(); val++) {
        norm += u[val] * v[val];
    }

    return norm;
}

void PCA::NormaliseVector(std::vector<double> &u) {
    double norm = std::sqrt(CalculateNorm(u, u));
    for (int val = 0; val < u.size(); val++) {
        u[val] /= norm;
    }
}

void PCA::ProjectVector(const std::vector<double> &u, const std::vector<double> &e, std::vector<double> &projected_vector) {
    double norm_1 = CalculateNorm(u, e);
    double norm_2 = CalculateNorm(e, e);
    for (int val = 0; val < e.size(); val++) {
        projected_vector[val] = e[val] * (norm_1 / norm_2);
    }
}

void PCA::MultiplyMatrices(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, std::vector<std::vector<double>> &c) {
    for (int row = 0; row < a.size(); row++) {
        for (int col = 0; col < b[0].size(); col++) {
            c[row][col] = 0.0;
            for (int element = 0; element < a[0].size(); element++) {
                c[row][col] += (a[row][element] * b[element][col]);
            }
        }
    }
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