#include "block_aca_bigwham.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr int kBlockSize = 3;
constexpr double kPi = 3.14159265358979323846;

struct Element {
  Eigen::Vector3d center;
  Eigen::Matrix3d local_basis;  // columns: local e1, e2 and normal
  double area;
};

// Demonstration kernel only.
//
// It returns one 3x3 interaction block directly; it does not access or build
// a global dense matrix.  Replace the body of this function with your actual
// triangular-element displacement/traction kernel.  For a fault BEM, column q
// should be the three-component response at obs caused by unit slip component
// q on src.
Eigen::Matrix3d computeKernelBlock(const Element& obs, const Element& src) {
  const Eigen::Vector3d r = obs.center - src.center;
  constexpr double regularization = 0.05;
  const double R2 = r.squaredNorm() + regularization * regularization;
  const double R = std::sqrt(R2);

  // A smooth Kelvin-like tensor is used so that this demo is self-contained.
  // The separated source/observation clusters make its matrix block low-rank.
  constexpr double poisson_ratio = 0.25;
  const Eigen::Matrix3d global_kernel =
      src.area / (16.0 * kPi * (1.0 - poisson_ratio)) *
      ((3.0 - 4.0 * poisson_ratio) / R * Eigen::Matrix3d::Identity() +
       (r * r.transpose()) / (R2 * R));

  // Rows: response components in the observation element's local basis.
  // Cols: source components in the source element's local basis.
  return obs.local_basis.transpose() * global_kernel * src.local_basis;
}

std::vector<Element> makeElements(int count, double x_offset) {
  std::vector<Element> elements;
  elements.reserve(static_cast<std::size_t>(count));

  for (int i = 0; i < count; ++i) {
    const int ix = i % 20;
    const int iy = i / 20;

    Element element;
    element.center = Eigen::Vector3d(
        x_offset + 0.4 * static_cast<double>(ix),
        0.4 * static_cast<double>(iy),
        0.05 * std::sin(0.2 * static_cast<double>(i)));
    element.local_basis = Eigen::Matrix3d::Identity();
    element.area = 0.16;
    elements.push_back(element);
  }
  return elements;
}

}  // namespace

int main() {
  using ACA = blockaca::BlockACA<kBlockSize, double>;
  using Vector = Eigen::VectorXd;

  // Two well-separated clusters correspond to one admissible H-matrix block.
  // ACA should normally be applied separately to each admissible far-field
  // cluster pair, not blindly to the whole BEM matrix including the near field.
  const std::vector<Element> observation_elements = makeElements(120, 0.0);
  const std::vector<Element> source_elements = makeElements(100, 40.0);

  std::size_t kernel_evaluation_count = 0;

  // Matrix-free interface: ACA calls this lambda only when it needs a block.
  ACA aca(
      static_cast<int>(observation_elements.size()),
      static_cast<int>(source_elements.size()),
      [&](int obs_id, int src_id) -> ACA::Block {
        ++kernel_evaluation_count;
        return computeKernelBlock(
            observation_elements[static_cast<std::size_t>(obs_id)],
            source_elements[static_cast<std::size_t>(src_id)]);
      });

  ACA::Options options;
  options.epsilon = 1.0e-6;
  options.max_block_rank = 30;
  options.initial_row = 0;
  options.pivot_rtol = 1.0e-14;

  const auto compressed = aca.compute(options);

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "Block rank                 : " << compressed.block_rank
            << '\n';
  std::cout << "Scalar rank upper bound    : "
            << compressed.scalar_rank_bound << '\n';
  std::cout << "Estimated relative update  : "
            << compressed.estimated_relative_update << '\n';
  std::cout << "Stop reason                : " << compressed.stop_reason
            << '\n';
  std::cout << "Kernel block evaluations   : "
            << kernel_evaluation_count << '\n';
  std::cout << "Full dense block count     : "
            << observation_elements.size() * source_elements.size()
            << '\n';
  std::cout << "A dimensions               : " << compressed.A.rows() << " x "
            << compressed.A.cols() << '\n';
  std::cout << "B dimensions               : " << compressed.B.rows() << " x "
            << compressed.B.cols() << '\n';

  // Three source degrees of freedom per element. In a fault problem these can
  // be strike-slip, dip-slip and opening components, provided that the kernel
  // uses exactly the same component order.
  Vector source_dofs = Vector::Zero(
      static_cast<Eigen::Index>(kBlockSize * source_elements.size()));
  for (std::size_t j = 0; j < source_elements.size(); ++j) {
    source_dofs(static_cast<Eigen::Index>(kBlockSize * j + 0)) = 1.0e-3;
    source_dofs(static_cast<Eigen::Index>(kBlockSize * j + 1)) =
        2.0e-4 * std::sin(0.1 * static_cast<double>(j));
    source_dofs(static_cast<Eigen::Index>(kBlockSize * j + 2)) = 0.0;
  }

  // response ~= A * (B * source_dofs); no dense matrix is reconstructed.
  const Vector response = compressed.matvec(source_dofs);

  std::cout << "\nFirst five three-component responses:\n";
  const int shown =
      std::min<int>(5, static_cast<int>(observation_elements.size()));
  for (int i = 0; i < shown; ++i) {
    std::cout << "obs " << i << " : "
              << response.segment<kBlockSize>(i * kBlockSize).transpose()
              << '\n';
  }

  return compressed.block_rank > 0 ? 0 : 1;
}

