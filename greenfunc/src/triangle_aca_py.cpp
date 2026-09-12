#include "block_aca_bigwham.hpp"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using namespace Eigen;


Matrix3d adaptive_integrate_traction(const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
                                     const Vector3d& slip, double nu, double mu,
                                     gsl_integration_workspace* w_outer, gsl_integration_workspace* w_inner,double eps_ratio=0.05);


void TDstressHS(const std::vector<double>& X,
                const std::vector<double>& Y,
                const std::vector<double>& Z,
                const std::vector<double>& P1,
                const std::vector<double>& P2,
                const std::vector<double>& P3,
                double Ss, double Ds, double Ts,
                double mu, double lambda_,
                std::vector<double>& Sxx, std::vector<double>& Syy, std::vector<double>& Szz,
                std::vector<double>& Sxy, std::vector<double>& Sxz, std::vector<double>& Syz,
                std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
                std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz);


void TDstressFS(const std::vector<double>& X,
                const std::vector<double>& Y,
                const std::vector<double>& Z,
                const std::vector<double>& P1,
                const std::vector<double>& P2,
                const std::vector<double>& P3,
                double Ss, double Ds, double Ts,
                double mu, double lambda_,
                std::vector<double>& Sxx, std::vector<double>& Syy, std::vector<double>& Szz,
                std::vector<double>& Sxy, std::vector<double>& Sxz, std::vector<double>& Syz,
                std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
                std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz);




Matrix3d rotate_matrix(
    const Vector3d& P1,
    const Vector3d& P2,
    const Vector3d& P3);


Matrix3d compute_traction_analytical_green_block(
    const Vector3d& obs_p1,
    const Vector3d& obs_p2,
    const Vector3d& obs_p3,

    const Vector3d& src_p1,
    const Vector3d& src_p2,
    const Vector3d& src_p3,

    double nu,
    double mu,
    bool use_half = true)
    {
        double lambda_ = 2.0 * mu * nu / (1.0 - 2.0 * nu);
        const Vector3d observation_point =
        (obs_p1 + obs_p2 + obs_p3) / 3.0;

    // Observation local-to-global rotation matrix
    const Matrix3d observation_rotation =
        rotate_matrix(
            obs_p1,
            obs_p2,
            obs_p3);

    // Observation normal in global coordinates
    const Vector3d observation_normal =observation_rotation.col(2);
        
    
    Matrix3d green_block =Matrix3d::Zero();
    
    size_t n = 1;
    std::vector<double> X(n), Y(n), Z(n);
    std::vector<double> Sxx(n), Syy(n), Szz(n);
    std::vector<double> Sxy(n), Sxz(n), Syz(n);
    std::vector<double> Exx(n), Eyy(n), Ezz(n);
    std::vector<double> Exy(n), Exz(n), Eyz(n);
    X[0]=observation_point[0];
    Y[0]=observation_point[1];
    Z[0]=observation_point[2];

    std::vector<double> P1 = {src_p1(0),src_p1(1),src_p1(2)};
    std::vector<double> P2 = {src_p2(0),src_p2(1),src_p2(2)};
    std::vector<double> P3 = {src_p3(0),src_p3(1),src_p3(2)};

    for (int slip_component = 0;
         slip_component < 3;
         ++slip_component)
        {
            Vector3d slip_local =
            Vector3d::Zero();

        slip_local(slip_component) = 1.0;

        if(use_half == true)
        {
            TDstressHS(X, Y, Z,
                    P1, P2, P3,
                    slip_local[0], slip_local[1], slip_local[2],
                    mu, lambda_,
                    Sxx, Syy, Szz, Sxy, Sxz, Syz,
                Exx, Eyy, Ezz, Exy, Exz, Eyz);
        }
        else
        {
            TDstressFS(X, Y, Z,
                    P1, P2, P3,
                    slip_local[0], slip_local[1], slip_local[2],
                    mu, lambda_,
                    Sxx, Syy, Szz, Sxy, Sxz, Syz,
                Exx, Eyy, Ezz, Exy, Exz, Eyz);
        }

        Matrix3d stress_global;

        stress_global <<
            Sxx[0], Sxy[0], Sxz[0],
            Sxy[0], Syy[0], Syz[0],
            Sxz[0], Syz[0], Szz[0];

        const Vector3d traction_global =stress_global * observation_normal;
            

        // Transform global traction into observation-local coordinates
        const Vector3d traction_local =observation_rotation.transpose()* traction_global;
            
        // Store as one column of the 3x3 Green matrix
        green_block.col(slip_component) =traction_local;

        }
    
        
        return green_block;
            
    }
    
    


Matrix3d compute_traction_green_block(
    const Vector3d& obs_p1,
    const Vector3d& obs_p2,
    const Vector3d& obs_p3,

    const Vector3d& src_p1,
    const Vector3d& src_p2,
    const Vector3d& src_p3,

    double nu,
    double mu,

    gsl_integration_workspace* w_outer,
    gsl_integration_workspace* w_inner,

    double eps_ratio = 0.05)
{
    // Observation point at the triangle centroid
    const Vector3d observation_point =
        (obs_p1 + obs_p2 + obs_p3) / 3.0;

    // Observation local-to-global rotation matrix
    const Matrix3d observation_rotation =
        rotate_matrix(
            obs_p1,
            obs_p2,
            obs_p3);

    // Observation normal in global coordinates
    const Vector3d observation_normal =
        observation_rotation.col(2);

    Matrix3d green_block =
        Matrix3d::Zero();

    for (int slip_component = 0;
         slip_component < 3;
         ++slip_component)
    {
        // Slip is already defined in the source local coordinate system
        Vector3d slip_local =
            Vector3d::Zero();

        slip_local(slip_component) = 1.0;

        // The slip input is not rotated
        const Matrix3d stress_global =
            adaptive_integrate_traction(
                observation_point,
                src_p1,
                src_p2,
                src_p3,
                slip_local,
                nu,
                mu,
                w_outer,
                w_inner,
                eps_ratio);

        // Global traction acting on the observation triangle
        const Vector3d traction_global =
            stress_global * observation_normal;

        // Transform global traction into observation-local coordinates
        const Vector3d traction_local =
            observation_rotation.transpose()
            * traction_global;

        // Store as one column of the 3x3 Green matrix
        green_block.col(slip_component) =
            traction_local;
    }

    return green_block;
}




// One constant triangular BIEM element.
struct Triangle {
  std::array<Eigen::Vector3d, 3> vertex;

  [[nodiscard]] Eigen::Vector3d center() const {
    return (vertex[0] + vertex[1] + vertex[2]) / 3.0;
  }

  [[nodiscard]] double area() const {
    return 0.5 * (vertex[1] - vertex[0])
                     .cross(vertex[2] - vertex[0])
                     .norm();
  }
};


std::vector<Triangle> numpy_to_triangles(
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& coords)
{
    if (coords.ndim() != 3 ||
        coords.shape(1) != 3 ||
        coords.shape(2) != 3)
    {
        throw std::invalid_argument(
            "triangle coordinates must have shape (N, 3, 3)"
        );
    }

    const py::ssize_t ntri = coords.shape(0);

    std::vector<Triangle> triangles;
    triangles.reserve(static_cast<std::size_t>(ntri));

    auto a = coords.unchecked<3>();

    for (py::ssize_t i = 0; i < ntri; ++i)
    {
        triangles.push_back({{
            Eigen::Vector3d(
                a(i, 0, 0),
                a(i, 0, 1),
                a(i, 0, 2)
            ),

            Eigen::Vector3d(
                a(i, 1, 0),
                a(i, 1, 1),
                a(i, 1, 2)
            ),

            Eigen::Vector3d(
                a(i, 2, 0),
                a(i, 2, 1),
                a(i, 2, 2)
            )
        }});
    }

    return triangles;
}



// ---------------------------------------------------------------------------
// Interface between triangle geometry, a variable Green function, and the
// unchanged BlockACA implementation.
//
// The Green function must be callable as:
//
// Eigen::Matrix3d green(int observation_id,
//                       int source_id,
//                       const Triangle& observation,
//                       const Triangle& source);
// ---------------------------------------------------------------------------
template <typename GreenFunction>
blockaca::Result<3, double> computeTriangleACA(
    const std::vector<Triangle>& observation_triangles,
    const std::vector<Triangle>& source_triangles,
    GreenFunction&& green_function,
    const blockaca::BlockACA<3, double>::Options& options = {}) {
  using ACA = blockaca::BlockACA<3, double>;

  if (observation_triangles.empty() || source_triangles.empty()) {
    throw std::invalid_argument("triangle lists must not be empty");
  }

  ACA aca(
      static_cast<int>(observation_triangles.size()),
      static_cast<int>(source_triangles.size()),
      [&](int i, int j) -> Eigen::Matrix3d {
        return std::invoke(
            green_function,
            i,
            j,
            observation_triangles.at(static_cast<std::size_t>(i)),
            source_triangles.at(static_cast<std::size_t>(j)));
      });

  return aca.compute(options);
}

// ---------------------------------------------------------------------------
// Test Green function 1.
// Replace only its body with your first triangle BIEM Green function.
// ---------------------------------------------------------------------------
Eigen::Matrix3d greenFunction1(
    int observation_id,
    int source_id,
    const Triangle& observation,
    const Triangle& source) {
  (void)observation_id;
  (void)source_id;

  const Eigen::Vector3d r = observation.center() - source.center();
  const double distance = r.norm();

  Eigen::Matrix3d coupling;
  coupling << 2.00, 0.20, 0.10,
              0.15, 1.50, 0.25,
              0.05, 0.20, 1.00;

  return source.area() / distance * coupling;
}

// Test Green function 2 with additional material parameters.
Eigen::Matrix3d greenFunction2(
    int observation_id,
    int source_id,
    const Triangle& observation,
    const Triangle& source,
    double shear_modulus,
    double poisson_ratio) {
  (void)observation_id;
  (void)source_id;

  const Eigen::Vector3d r = observation.center() - source.center();
  const double distance_squared = r.squaredNorm();

  Eigen::Matrix3d coupling;
  coupling << 1.00, poisson_ratio, 0.05,
              poisson_ratio, 1.20, 0.10,
              0.05, 0.10, 0.80;

  return shear_modulus * source.area() / distance_squared * coupling;
}

// Self-contained test coordinates. Replace this with your triangle mesh input.
std::vector<Triangle> makeTriangles(int count, double x_offset) {
  std::vector<Triangle> triangles;
  triangles.reserve(static_cast<std::size_t>(count));

  constexpr double side = 0.20;
  for (int index = 0; index < count; ++index) {
    const int ix = index % 10;
    const int iy = index / 10;
    const double x = x_offset + 0.30 * static_cast<double>(ix);
    const double y = 0.30 * static_cast<double>(iy);
    const double z = 0.02 * std::sin(0.10 * static_cast<double>(index));

    triangles.push_back({{
        Eigen::Vector3d(x, y, z),
        Eigen::Vector3d(x + side, y, z),
        Eigen::Vector3d(x, y + side, z),
    }});
  }

  return triangles;
}

Eigen::Matrix3d approximateBlock(
    const blockaca::Result<3, double>& result,
    int observation_id,
    int source_id) {
  return result.A.middleRows(3 * observation_id, 3) *
         result.B.middleCols(3 * source_id, 3);
}

void printResult(const char* name,
                 const blockaca::Result<3, double>& result) {
  std::cout << name << '\n'
            << "  block rank   : " << result.block_rank << '\n'
            << "  scalar bound : " << result.scalar_rank_bound << '\n'
            << "  A dimensions : " << result.A.rows() << " x "
            << result.A.cols() << '\n'
            << "  B dimensions : " << result.B.rows() << " x "
            << result.B.cols() << '\n'
            << "  update error : " << result.estimated_relative_update << '\n'
            << "  stop reason  : " << result.stop_reason << "\n\n";
}




// blockaca::Result<3, double> compute_triangle_aca_python(
//     const std::vector<Triangle>& observation_triangles,
//     const std::vector<Triangle>& source_triangles,
//     double nu,
//     double mu,
//     double eps_ratio,
//     const blockaca::BlockACA<3, double>::Options& options)
// {
//     gsl_integration_workspace* w_outer =
//         gsl_integration_workspace_alloc(1000);

//     gsl_integration_workspace* w_inner =
//         gsl_integration_workspace_alloc(1000);

//     const auto compressed_green = computeTriangleACA(
//         observation_triangles,
//         source_triangles,

//         [nu, mu, w_outer, w_inner, eps_ratio]
//         (
//             int observation_id,
//             int source_id,
//             const Triangle& observation,
//             const Triangle& source
//         ) -> Eigen::Matrix3d
//         {
//             return compute_traction_green_block(
//                 observation.vertex[0],
//                 observation.vertex[1],
//                 observation.vertex[2],

//                 source.vertex[0],
//                 source.vertex[1],
//                 source.vertex[2],

//                 nu,
//                 mu,

//                 w_outer,
//                 w_inner,

//                 eps_ratio
//             );
//         },

//         options
//     );

//     gsl_integration_workspace_free(w_outer);
//     gsl_integration_workspace_free(w_inner);

//     return compressed_green;
// }



// blockaca::Result<3, double> compute_triangle_aca_analytical_python(
//     const std::vector<Triangle>& observation_triangles,
//     const std::vector<Triangle>& source_triangles,
//     double nu,
//     double mu,
//     bool use_half,
//     const blockaca::BlockACA<3, double>::Options& options)
// {
//     const auto compressed_green = computeTriangleACA(
//         observation_triangles,
//         source_triangles,

//         [nu, mu,use_half]
//         (
//             int observation_id,
//             int source_id,
//             const Triangle& observation,
//             const Triangle& source
//         ) -> Eigen::Matrix3d
//         {
//             return compute_traction_analytical_green_block(
//                 observation.vertex[0],
//                 observation.vertex[1],
//                 observation.vertex[2],

//                 source.vertex[0],
//                 source.vertex[1],
//                 source.vertex[2],

//                 nu,
//                 mu,
//                 use_half
//             );
//         },

//         options
//     );

//     return compressed_green;
// }

blockaca::Result<3, double>
compute_triangle_aca_python0(
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& observation_coordinates,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& source_coordinates,
    double nu,
    double mu,
    double eps_ratio,
    const blockaca::BlockACA<3, double>::Options& options)
{
    const std::vector<Triangle> observation_triangles =
        numpy_to_triangles(observation_coordinates);

    const std::vector<Triangle> source_triangles =
        numpy_to_triangles(source_coordinates);

    gsl_integration_workspace* w_outer =
        // Must be at least the outer QAG/QAGS limit used by
        // integrate_duffy_subtriangle in dislocation_lib_duffy.cpp.
        gsl_integration_workspace_alloc(5000);

    gsl_integration_workspace* w_inner =
        gsl_integration_workspace_alloc(1000);

    const auto compressed_green = computeTriangleACA(
        observation_triangles,
        source_triangles,

        [nu, mu, w_outer, w_inner, eps_ratio]
        (
            int observation_id,
            int source_id,
            const Triangle& observation,
            const Triangle& source
        ) -> Eigen::Matrix3d
        {
            return compute_traction_green_block(
                observation.vertex[0],
                observation.vertex[1],
                observation.vertex[2],

                source.vertex[0],
                source.vertex[1],
                source.vertex[2],

                nu,
                mu,

                w_outer,
                w_inner,

                eps_ratio
            );
        },

        options
    );

    gsl_integration_workspace_free(w_outer);
    gsl_integration_workspace_free(w_inner);

    return compressed_green;
}




blockaca::Result<3, double>
compute_triangle_aca_python(
    const py::array_t<
        double,
        py::array::c_style | py::array::forcecast
    >& observation_coordinates,

    const py::array_t<
        double,
        py::array::c_style | py::array::forcecast
    >& source_coordinates,

    double nu,
    double mu,

    // Accept Python float, list, tuple, or NumPy array
    const py::object& eps_ratio_input,

    const blockaca::BlockACA<3, double>::Options& options)
{
    const std::vector<Triangle> observation_triangles =
        numpy_to_triangles(observation_coordinates);

    const std::vector<Triangle> source_triangles =
        numpy_to_triangles(source_coordinates);

    const std::size_t number_of_sources = source_triangles.size();

    /*
     * Convert eps_ratio_input to a NumPy array.
     *
     * Supported inputs:
     *     0.05
     *     [0.01, 0.02, 0.03]
     *     np.array([...])
     */
    py::array_t<
        double,
        py::array::c_style | py::array::forcecast
    > eps_array =
        py::array_t<
            double,
            py::array::c_style | py::array::forcecast
        >::ensure(eps_ratio_input);

    if (!eps_array) {
        throw py::type_error(
            "eps_ratio must be a scalar or a one-dimensional array."
        );
    }

    Eigen::VectorXd eps_ratio(
        static_cast<Eigen::Index>(number_of_sources)
    );

    if (eps_array.ndim() == 0) {
        // Python/NumPy scalar: broadcast to every source
        const double scalar_eps =
            *static_cast<const double*>(eps_array.data());

        eps_ratio.setConstant(scalar_eps);
    }
    else if (eps_array.ndim() == 1) {
        const py::ssize_t input_size = eps_array.shape(0);
        const double* eps_data = eps_array.data();

        if (input_size == 1) {
            // A one-element vector is also treated as a scalar
            eps_ratio.setConstant(eps_data[0]);
        }
        else if (
            input_size ==
            static_cast<py::ssize_t>(number_of_sources)
        ) {
            // One eps_ratio for each source triangle
            for (std::size_t i = 0; i < number_of_sources; ++i) {
                eps_ratio(static_cast<Eigen::Index>(i)) =
                    eps_data[i];
            }
        }
        else {
            throw py::value_error(
                "eps_ratio must be a scalar, a one-element vector, "
                "or a vector whose length equals the number of "
                "source triangles."
            );
        }
    }
    else {
        throw py::value_error(
            "eps_ratio must be a scalar or a one-dimensional array."
        );
    }

    gsl_integration_workspace* w_outer =
        // Must be at least the outer QAG/QAGS limit used by
        // integrate_duffy_subtriangle in dislocation_lib_duffy.cpp.
        gsl_integration_workspace_alloc(5000);

    gsl_integration_workspace* w_inner =
        gsl_integration_workspace_alloc(1000);

    if (w_outer == nullptr || w_inner == nullptr) {
        if (w_outer != nullptr) {
            gsl_integration_workspace_free(w_outer);
        }

        if (w_inner != nullptr) {
            gsl_integration_workspace_free(w_inner);
        }

        throw std::runtime_error(
            "Failed to allocate GSL integration workspaces."
        );
    }

    try {
        const auto compressed_green = computeTriangleACA(
            observation_triangles,
            source_triangles,

            [nu, mu, w_outer, w_inner, eps_ratio]
            (
                int observation_id,
                int source_id,
                const Triangle& observation,
                const Triangle& source
            ) -> Eigen::Matrix3d
            {
                if (
                    source_id < 0 ||
                    source_id >= eps_ratio.size()
                ) {
                    throw std::out_of_range(
                        "source_id is outside the eps_ratio range."
                    );
                }

                // Choose eps_ratio corresponding to this source
                const double source_eps_ratio =
                    eps_ratio(source_id);

                return compute_traction_green_block(
                    observation.vertex[0],
                    observation.vertex[1],
                    observation.vertex[2],

                    source.vertex[0],
                    source.vertex[1],
                    source.vertex[2],

                    nu,
                    mu,

                    w_outer,
                    w_inner,

                    source_eps_ratio
                );
            },

            options
        );

        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);

        return compressed_green;
    }
    catch (...) {
        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
        throw;
    }
}


blockaca::Result<3, double>
compute_triangle_aca_analytical_python(
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& observation_coordinates,
    const py::array_t<double,
        py::array::c_style | py::array::forcecast>& source_coordinates,
    double nu,
    double mu,
    bool use_half,
    const blockaca::BlockACA<3, double>::Options& options)
{
    const std::vector<Triangle> observation_triangles =
        numpy_to_triangles(observation_coordinates);

    const std::vector<Triangle> source_triangles =
        numpy_to_triangles(source_coordinates);


    const auto compressed_green = computeTriangleACA(
        observation_triangles,
        source_triangles,

        [nu, mu, use_half]
        (
            int observation_id,
            int source_id,
            const Triangle& observation,
            const Triangle& source
        ) -> Eigen::Matrix3d
        {
            return compute_traction_analytical_green_block(
                observation.vertex[0],
                observation.vertex[1],
                observation.vertex[2],

                source.vertex[0],
                source.vertex[1],
                source.vertex[2],

                nu,
                mu,
                use_half
            );
        },

        options
    );

    return compressed_green;
}




PYBIND11_MODULE(block_aca_lib, m)
{
    // GSL's default handler aborts the whole MPI worker on recoverable
    // integration failures. The integration code checks status values and
    // provides bounded fallback quadrature, so return errors to the caller.
    gsl_set_error_handler_off();

    m.doc() = "Block ACA interface for triangular BIEM elements";

    using ACA = blockaca::BlockACA<3, double>;
    using Result = blockaca::Result<3, double>;

    py::class_<ACA::Options>(m, "ACAOptions")
        .def(py::init<>())
        .def_readwrite("epsilon", &ACA::Options::epsilon)
        .def_readwrite("max_block_rank", &ACA::Options::max_block_rank)
        .def_readwrite("initial_row", &ACA::Options::initial_row)
        .def_readwrite("pivot_rtol", &ACA::Options::pivot_rtol);

    py::class_<Result>(m, "ACAResult")
        .def_readonly("A", &Result::A)
        .def_readonly("B", &Result::B)
        .def_readonly("block_rank", &Result::block_rank)
        .def("matvec", &Result::matvec);

    m.def(
        "compute_triangle_aca",
        &compute_triangle_aca_python,
        py::arg("observation_triangles"),
        py::arg("source_triangles"),
        py::arg("nu"),
        py::arg("mu"),
        py::arg("eps_ratio") = 0.05,
        py::arg("options")
    );

    m.def(
        "compute_triangle_aca_analytical",
        &compute_triangle_aca_analytical_python,
        py::arg("observation_triangles"),
        py::arg("source_triangles"),
        py::arg("nu"),
        py::arg("mu"),
        py::arg("use_half"),
        py::arg("options")
    );
}

// PYBIND11_MODULE(block_aca_lib, m)
// {
//     m.doc() = "Block ACA interface for triangular BIEM elements";

//     // Triangle
//     py::class_<Triangle>(m, "Triangle")
//         .def(py::init<>())
//         .def_readwrite("vertex", &Triangle::vertex);

//     // ACA options
//     using ACA = blockaca::BlockACA<3, double>;
//     using Result = blockaca::Result<3, double>;

//     py::class_<ACA::Options>(m, "ACAOptions")
//         .def(py::init<>())
//         .def_readwrite("epsilon", &ACA::Options::epsilon)
//         .def_readwrite("max_block_rank", &ACA::Options::max_block_rank)
//         .def_readwrite("initial_row", &ACA::Options::initial_row)
//         .def_readwrite("pivot_rtol", &ACA::Options::pivot_rtol);

//     // ACA result
//     py::class_<Result>(m, "ACAResult")
//         .def_readonly("A", &Result::A)
//         .def_readonly("B", &Result::B)
//         .def_readonly("block_rank", &Result::block_rank)
//         .def("matvec", &Result::matvec);

//     // Main Python interface
//     m.def(
//         "compute_triangle_aca",
//         &compute_triangle_aca_python,
//         py::arg("observation_triangles"),
//         py::arg("source_triangles"),
//         py::arg("nu"),
//         py::arg("mu"),
//         py::arg("eps_ratio") = 0.05,
//         py::arg("options")
//     );
//     m.def(
//         "compute_triangle_aca_analytical",
//         &compute_triangle_aca_analytical_python,
//         py::arg("observation_triangles"),
//         py::arg("source_triangles"),
//         py::arg("nu"),
//         py::arg("mu"),
//         py::arg("use_half"),
//         py::arg("options")
//     );
// }
