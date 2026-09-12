#pragma once

// Block adaptive cross approximation inspired by BigWham.
//
// The matrix is treated as n_block_rows x n_block_cols blocks, each P x P:
//
//   M ~= A B,
//   A_k = R(:, j_k),
//   B_k = R(i_k, j_k)^{-1} R(i_k, :).
//
// Unlike the reference code, this version solves P_k B_k = R(i_k, :) rather
// than explicitly forming P_k^{-1}.  It is header-only and uses Eigen.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blockaca {

template <int P, typename Scalar = double>
struct Result {
  static_assert(P > 0, "P must be positive");

  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  Matrix A;
  Matrix B;
  int block_rank = 0;
  int scalar_rank_bound = 0;
  std::vector<int> pivot_rows;
  std::vector<int> pivot_cols;
  bool converged = false;
  std::string stop_reason;
  double estimated_relative_update =
      std::numeric_limits<double>::infinity();

  [[nodiscard]] Matrix reconstruct() const { return A * B; }

  [[nodiscard]] Vector matvec(const Eigen::Ref<const Vector>& x) const {
    if (x.size() != B.cols()) {
      throw std::invalid_argument("matvec: incompatible vector size");
    }
    return A * (B * x);
  }
};

template <int P, typename Scalar = double>
class BlockACA {
 public:
  static_assert(P > 0, "P must be positive");

  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using Block = Eigen::Matrix<Scalar, P, P>;
  using BlockGenerator = std::function<Block(int, int)>;

  struct Options {
    double epsilon = 1.0e-6;
    int max_block_rank = -1;  // -1 means min(n_block_rows, n_block_cols)
    int initial_row = 0;
    double pivot_rtol = 1.0e-14;
  };

  BlockACA(int n_block_rows, int n_block_cols, BlockGenerator generator)
      : n0_(n_block_rows), n1_(n_block_cols), generator_(std::move(generator)) {
    if (n0_ <= 0 || n1_ <= 0) {
      throw std::invalid_argument("block dimensions must be positive");
    }
    if (!generator_) {
      throw std::invalid_argument("block generator is empty");
    }
  }

  [[nodiscard]] Result<P, Scalar> compute(const Options& options = {}) const {
    validateOptions(options);

    const int rank_limit =
        options.max_block_rank < 0
            ? std::min(n0_, n1_)
            : std::min({options.max_block_rank, n0_, n1_});

    Result<P, Scalar> result;
    result.A.resize(n0_ * P, 0);
    result.B.resize(0, n1_ * P);
    result.stop_reason = "maximum block rank reached";

    std::vector<bool> used_rows(static_cast<std::size_t>(n0_), false);
    std::vector<bool> used_cols(static_cast<std::size_t>(n1_), false);
    int i_search = options.initial_row;

    for (int iteration = 0; iteration < rank_limit; ++iteration) {
      const Matrix row = residualRow(result.A, result.B, i_search);
      const int j_search =
          choosePivotColumn(row, used_cols, options.pivot_rtol);
      if (j_search < 0) {
        result.stop_reason = "no nonsingular unused pivot block";
        break;
      }

      const Block pivot = row.template middleCols<P>(j_search * P);
      const Matrix column = residualColumn(result.A, result.B, j_search);

      // COD is robust for the very small P x P pivot block and lets us reject
      // a singular pivot. Solving is preferable to pivot.inverse().
      Eigen::CompleteOrthogonalDecomposition<Block> decomposition(pivot);
      decomposition.setThreshold(options.pivot_rtol);
      if (decomposition.rank() < P) {
        result.stop_reason = "singular pivot block";
        break;
      }
      const Matrix Bk = decomposition.solve(row);
      const Matrix Ak = column;
      if (!Ak.allFinite() || !Bk.allFinite()) {
        result.stop_reason = "non-finite low-rank factor";
        break;
      }

      appendFactors(Ak, Bk, result.A, result.B);
      used_rows[static_cast<std::size_t>(i_search)] = true;
      used_cols[static_cast<std::size_t>(j_search)] = true;
      result.pivot_rows.push_back(i_search);
      result.pivot_cols.push_back(j_search);

      const double update_sq = productFrobeniusSquared(Ak, Bk);
      const double approximation_sq =
          productFrobeniusSquared(result.A, result.B);
      result.estimated_relative_update =
          approximation_sq > 0.0
              ? std::sqrt(update_sq / approximation_sq)
              : 0.0;

      if (result.estimated_relative_update <= options.epsilon) {
        result.stop_reason = "relative update tolerance reached";
        break;
      }

      const int next_i = chooseNextRow(column, used_rows);
      if (next_i < 0) {
        result.stop_reason = "no unused residual block row";
        break;
      }
      i_search = next_i;
    }

    result.block_rank = static_cast<int>(result.pivot_rows.size());
    result.scalar_rank_bound = result.block_rank * P;
    result.converged =
        result.stop_reason == "relative update tolerance reached" ||
        result.stop_reason == "no nonsingular unused pivot block" ||
        result.stop_reason == "no unused residual block row";
    return result;
  }

  // Convenience overload for an already assembled dense matrix.
  static Result<P, Scalar> computeDense(
      const Eigen::Ref<const Matrix>& dense, const Options& options = {}) {
    if (dense.rows() % P != 0 || dense.cols() % P != 0) {
      throw std::invalid_argument(
          "dense matrix dimensions must be divisible by P");
    }
    const int n0 = static_cast<int>(dense.rows() / P);
    const int n1 = static_cast<int>(dense.cols() / P);
    BlockACA aca(n0, n1, [&dense](int i, int j) -> Block {
      return dense.template block<P, P>(i * P, j * P);
    });
    return aca.compute(options);
  }

 private:
  int n0_;
  int n1_;
  BlockGenerator generator_;

  void validateOptions(const Options& options) const {
    if (!(options.epsilon > 0.0)) {
      throw std::invalid_argument("epsilon must be positive");
    }
    if (options.pivot_rtol < 0.0) {
      throw std::invalid_argument("pivot_rtol must be nonnegative");
    }
    if (options.initial_row < 0 || options.initial_row >= n0_) {
      throw std::invalid_argument("initial_row is outside block-row range");
    }
    if (options.max_block_rank == 0) {
      throw std::invalid_argument("max_block_rank must be positive or -1");
    }
  }

  [[nodiscard]] Matrix residualRow(const Matrix& A, const Matrix& B,
                                   int i) const {
    Matrix row(P, n1_ * P);
    for (int j = 0; j < n1_; ++j) {
      row.template middleCols<P>(j * P) = generator_(i, j);
    }
    if (A.cols() > 0) {
      row.noalias() -= A.middleRows(i * P, P) * B;
    }
    return row;
  }

  [[nodiscard]] Matrix residualColumn(const Matrix& A, const Matrix& B,
                                      int j) const {
    Matrix column(n0_ * P, P);
    for (int i = 0; i < n0_; ++i) {
      column.template middleRows<P>(i * P) = generator_(i, j);
    }
    if (A.cols() > 0) {
      column.noalias() -= A * B.middleCols(j * P, P);
    }
    return column;
  }

  [[nodiscard]] int choosePivotColumn(const Matrix& row,
                                      const std::vector<bool>& used,
                                      double pivot_rtol) const {
    int best_j = -1;
    double best_smallest_singular_value = -1.0;

    for (int j = 0; j < n1_; ++j) {
      if (used[static_cast<std::size_t>(j)]) {
        continue;
      }
      const Block candidate = row.template middleCols<P>(j * P);
      Eigen::JacobiSVD<Block> svd(candidate, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
      const double sigma_min =
          static_cast<double>(svd.singularValues()(P - 1));
      if (std::isfinite(sigma_min) &&
          sigma_min > best_smallest_singular_value) {
        best_smallest_singular_value = sigma_min;
        best_j = j;
      }
    }

    const double scale =
        std::max(static_cast<double>(row.norm()),
                 std::numeric_limits<double>::min());
    if (best_j < 0 ||
        best_smallest_singular_value <= pivot_rtol * scale) {
      return -1;
    }
    return best_j;
  }

  [[nodiscard]] int chooseNextRow(const Matrix& column,
                                  const std::vector<bool>& used) const {
    int best_i = -1;
    double best_norm = 0.0;
    for (int i = 0; i < n0_; ++i) {
      if (used[static_cast<std::size_t>(i)]) {
        continue;
      }
      const double norm =
          static_cast<double>(column.middleRows(i * P, P).norm());
      if (std::isfinite(norm) && norm > best_norm) {
        best_norm = norm;
        best_i = i;
      }
    }
    return best_i;
  }

  static void appendFactors(const Matrix& Ak, const Matrix& Bk, Matrix& A,
                            Matrix& B) {
    const Eigen::Index old_rank = A.cols();
    Matrix new_A(A.rows(), old_rank + P);
    Matrix new_B(old_rank + P, B.cols());
    if (old_rank > 0) {
      new_A.leftCols(old_rank) = A;
      new_B.topRows(old_rank) = B;
    }
    new_A.template rightCols<P>() = Ak;
    new_B.template bottomRows<P>() = Bk;
    A.swap(new_A);
    B.swap(new_B);
  }

  // ||A B||_F^2 = Re tr[(A* A)(B B*)], without forming the large product AB.
  [[nodiscard]] static double productFrobeniusSquared(const Matrix& A,
                                                      const Matrix& B) {
    if (A.cols() == 0) {
      return 0.0;
    }
    const Matrix gram_A = A.adjoint() * A;
    const Matrix gram_B = B * B.adjoint();
    using std::real;
    const double value = static_cast<double>(real((gram_A * gram_B).trace()));
    return std::max(0.0, value);
  }
};

}  // namespace blockaca
