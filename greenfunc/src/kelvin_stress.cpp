
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <gsl/gsl_integration.h>
#include<fstream>
#include <iomanip>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gsl/gsl_errno.h>
#include <pybind11/numpy.h> //  array_t

namespace py = pybind11;
using namespace Eigen;

// Struct to pass parameters to GSL integrands
struct IntegrationParams {
    Vector3d obs_pt;
    Vector3d p1, v1, v2;
    Vector3d F;
    double lam, mu, nu,eps;
    int comp_i, comp_j;
    double u_val; // <--- Added this to track the outer integral state
};


Vector3d get_regularized_displacement(const Vector3d& x, const Vector3d& F, double lam, double mu, double eps = 1e-3) {
    double r2 = x.squaredNorm();
    double R2 = r2 + eps * eps;
    double R = std::sqrt(R2);
    double R3 = R2 * R;

    // 
    double C1 = (2.0 * r2 + 3.0 * eps * eps) / (8.0 * M_PI * mu * std::pow(R2, 1.5));
    
    // 
    double C2 = (lam + mu) / (8.0 * M_PI * mu * (lam + 2.0 * mu));
    
    //  f_j * x_j ()
    double fjxj = F.dot(x);
    
    //  u_i
    Vector3d u = C1 * F - C2 * (F / R - (fjxj * x) / (R3));
    
    return u;
}



// --- Core Kelvin Regularized Kernel ---
Matrix3d get_regularized_stress(const Vector3d& r, const Vector3d& F, double nu, double eps = 1e-3) {
    double r2 = r.squaredNorm();
    double R2 = r2 + eps * eps;
    double R = std::sqrt(R2);
    double R3 = R * R * R;
    //double R5 = R2 * R3;

    double C = 1.0 / (8.0 * M_PI * (1.0 - nu) * R3);
    double term1_const = (1.0 - 2.0 * nu) * (r2 + 1.5 * eps * eps) / R2;

    Matrix3d sigma = Matrix3d::Zero();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i+1; ++j) {
            double stress_val = 0.0;
            for (int k = 0; k < 3; ++k) {
                double dik = (i == k) ? 1.0 : 0.0;
                double djk = (j == k) ? 1.0 : 0.0;
                double dij = (i == j) ? 1.0 : 0.0;

                double part1 = term1_const * (dik * r(j) + djk * r(i) - dij * r(k));
                double part2 = 3.0 * r(i) * r(j) * r(k) / R2;
                double corr_term = (dik * r(j) + djk * r(i) + dij * r(k) * (nu / (1.0 - 2.0 * nu)));
                double part3 = -1.5 * (eps * eps / R2) * corr_term;

                stress_val += C * (part1 + part2 + part3) * F(k);
            }
            sigma(i, j) = -stress_val;
            if(i!=j)
            {
                sigma(j, i) = -stress_val;
            }
            
        }
    }
    return sigma;
}

// --- GSL Integration Functions ---
// Struct helper for nested integration
struct OuterParams {
    IntegrationParams* base;
    gsl_integration_workspace* w_inner;
};

// --- GSL ： ---
double disp_inner_integrand(double v, void* params) {
    auto* p = static_cast<IntegrationParams*>(params);
    Vector3d source_pt = p->p1 + (p->u_val * p->v1) + (v * p->v2);
    Vector3d r = p->obs_pt - source_pt;
    
    // 
    Vector3d disp = get_regularized_displacement(r, p->F, p->lam, p->mu, p->eps);
    return disp(p->comp_i); //  u_x, u_y  u_z 
}

//  outer_integrand ()
double outer_disp_integrand(double u, void* params) {
    auto* op = static_cast<OuterParams*>(params);
    op->base->u_val = u; 
    gsl_function F_inner;
    F_inner.function = &disp_inner_integrand;
    F_inner.params = op->base;

    double result, error;
    gsl_integration_qags(&F_inner, 0, 1.0 - u, 0, 1e-4, 1000, op->w_inner, &result, &error);
    return result;
}

// Inner integral (v direction)
double inner_integrand(double v, void* params) {
    auto* p = static_cast<IntegrationParams*>(params);
    
    // P = p1 + u*v1 + v*v2
    Vector3d source_pt = p->p1 + (p->u_val * p->v1) + (v * p->v2);
    Vector3d r = p->obs_pt - source_pt;
    
    // Calculate stress at this specific point
    Matrix3d sigma = get_regularized_stress(r, p->F, p->nu,p->eps);
    return sigma(p->comp_i, p->comp_j);
}

// Outer integral (u direction)
double outer_integrand(double u, void* params) {
    auto* op = static_cast<OuterParams*>(params);
    
    // Update the u coordinate for the inner integration
    op->base->u_val = u; 

    gsl_function F_inner;
    F_inner.function = &inner_integrand;
    F_inner.params = op->base;

    double result, error;
    // Inner limit: 0 to 1-u (The triangle domain)
    gsl_integration_qags(&F_inner, 0, 1.0 - u, 0, 1e-4, 1000, op->w_inner, &result, &error);
    return result;
}



Vector3d integrate_triangle_gauss_disp(const Vector3d& obs, const Vector3d& v1, const Vector3d& v2, const Vector3d& v3, 
                                     const Vector3d& F, double lam, double mu, double eps = 1e-3) {
    // 7
    std::vector<double> w = {0.225, 0.125939, 0.125939, 0.125939, 0.132394, 0.132394, 0.132394};
    std::vector<Vector3d> b = {
        {1.0/3, 1.0/3, 1.0/3}, {0.797427, 0.101287, 0.101287}, {0.101287, 0.797427, 0.101287},
        {0.101287, 0.101287, 0.797427}, {0.059716, 0.470142, 0.470142}, {0.470142, 0.059716, 0.470142},
        {0.470142, 0.470142, 0.059716}
    };

    Vector3d total_disp = Vector3d::Zero();
    double area = 0.5 * (v2 - v1).cross(v3 - v1).norm();

    for (int k = 0; k < 7; ++k) {
        Vector3d qp = b[k](0) * v1 + b[k](1) * v2 + b[k](2) * v3;
        total_disp += w[k] * get_regularized_displacement(obs - qp, F, lam, mu, eps);
    }

    return total_disp * area;
}

// --- Quadrature & Logic ---

Matrix3d integrate_triangle_gauss(const Vector3d& obs, const Vector3d& v1, const Vector3d& v2, const Vector3d& v3, 
                                 const Vector3d& F, double lam, double mu) {
    // 7-point Gauss weights and barycentric coordinates
    std::vector<double> w = {0.225, 0.125939, 0.125939, 0.125939, 0.132394, 0.132394, 0.132394};
    std::vector<Vector3d> b = {
        {1.0/3, 1.0/3, 1.0/3}, {0.797427, 0.101287, 0.101287}, {0.101287, 0.797427, 0.101287},
        {0.101287, 0.101287, 0.797427}, {0.059716, 0.470142, 0.470142}, {0.470142, 0.059716, 0.470142},
        {0.470142, 0.470142, 0.059716}
    };

    Matrix3d total_sigma = Matrix3d::Zero();
    double nu = lam / (2.0 * (lam + mu));
    double area = 0.5 * (v2 - v1).cross(v3 - v1).norm();

    for (int k = 0; k < 7; ++k) {
        Vector3d qp = b[k](0) * v1 + b[k](1) * v2 + b[k](2) * v3;
        total_sigma += w[k] * get_regularized_stress(obs - qp, F, nu);
    }

    return total_sigma * area;
}


Vector3d adaptive_integrate_disp(const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
                                 const Vector3d& F, double lam, double mu,
                                 gsl_integration_workspace* w_outer, 
                                 gsl_integration_workspace* w_inner) 
{
    double side1 = (p2 - p1).norm();
    double side2 = (p3 - p2).norm();
    double side3 = (p1 - p3).norm();
    double max_side = std::max({side1, side2, side3});
    double eps = 1.0/20.0 * max_side;
    Vector3d centroid = (p1 + p2 + p3) / 3.0;
    double dist = (obs - centroid).norm();

    // 1. ： ()
    if (dist > 5.0 * max_side) {
        return integrate_triangle_gauss_disp(obs, p1, p2, p3, F, lam, mu, eps);
    } 
    
    // 2. ：GSL  ()
    Vector3d total_disp = Vector3d::Zero();
    double jacobian = (p2 - p1).cross(p3 - p1).norm();

    IntegrationParams base_params;
    base_params.obs_pt = obs; base_params.p1 = p1; 
    base_params.v1 = p2 - p1; base_params.v2 = p3 - p1;
    base_params.F = F; base_params.lam = lam; base_params.mu = mu;
    base_params.eps = eps;

    for (int i = 0; i < 3; ++i) { //  u_x, u_y, u_z 
        base_params.comp_i = i;
        OuterParams op = {&base_params, w_inner};
        gsl_function F_outer;
        F_outer.function = &outer_disp_integrand;
        F_outer.params = &op;

        double res, err;
        gsl_integration_qags(&F_outer, 0, 1.0, 0, 1e-4, 1000, w_outer, &res, &err);
        total_disp(i) = res * jacobian;
    }
    return total_disp;
}


Matrix3d adaptive_integrate(const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
                            const Vector3d& F, double lam, double mu) {
    
    double side1 = (p2 - p1).norm();
    double side2 = (p3 - p2).norm();
    double side3 = (p1 - p3).norm();
    double max_side = std::max({side1, side2, side3});
    double eps=1.0/20.0*max_side;
    Vector3d centroid = (p1 + p2 + p3) / 3.0;
    double dist = (obs - centroid).norm();

    if (dist > 5.0 * max_side) {
        return integrate_triangle_gauss(obs, p1, p2, p3, F, lam, mu);
    } else {
        // GSL Numerical Integration (simulating dblquad)
        Matrix3d total_sigma = Matrix3d::Zero();
        double nu = lam / (2.0 * (lam + mu));
        double jacobian = (p2 - p1).cross(p3 - p1).norm();

        gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(1000);
        gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000);
        
        IntegrationParams base_params;
        base_params.obs_pt = obs; base_params.p1 = p1; 
        base_params.v1 = p2 - p1; base_params.v2 = p3 - p1;
        base_params.F = F; base_params.nu = nu;
        base_params.eps=eps;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                base_params.comp_i = i;
                base_params.comp_j = j;

                OuterParams op = {&base_params, w_inner};
                gsl_function F_outer;
                F_outer.function = &outer_integrand;
                F_outer.params = &op;

                double res, err;
                gsl_integration_qags(&F_outer, 0, 1.0, 0, 1e-4, 1000, w_outer, &res, &err);
                total_sigma(i, j) = res * jacobian;
            }
        }

        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
        return total_sigma;
    }
}


void save_stress_results(const std::string& filename, const std::vector<Matrix3d>& S_list) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    //  (Tab )
    outFile << "Index\tSxx\tSxy\tSxz\tSyy\tSyz\tSzz" << std::endl;

    //  ( 10 )
    outFile << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < S_list.size(); ++i) {
        const Matrix3d& S = S_list[i];
        outFile << i << "\t"
                << S(0, 0) << "\t"  // Sxx
                << S(0, 1) << "\t"  // Sxy
                << S(0, 2) << "\t"  // Sxz
                << S(1, 1) << "\t"  // Syy
                << S(1, 2) << "\t"  // Syz
                << S(2, 2)          // Szz
                << std::endl;
    }

    outFile.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main() {
    // Vector3d P1(0, 0, 0), P2(1, 0, 0), P3(0, 1, 0);
    // Vector3d F(1, 0, 0);
    // double mu = 20e8, nu = 0.3;
    // double lam = 2 * mu * nu / (1 - 2 * nu);

    // Vector3d obs(0.3, 0.4, 1.01);
    
    // Matrix3d S = integrate_triangle_gsl(obs, P1, P2, P3, F, lam, mu);

    // std::cout << "Integrated Stress Tensor:\n" << S << std::endl;
    // return 0;
    double mu = 20e8, nu = 0.3;
    double lam = 2 * mu * nu / (1 - 2 * nu);
    Vector3d F_vec(0.0, 0.0, 1.0);

    // 
    Vector3d P1(0.0, 0.0, -5.0);
    Vector3d P2(0.0, 1.0, -5.0);
    Vector3d P3(1.0, 0.0, -5.0);

    std::cout << "Starting integration loop..." << std::endl;
    std::vector<Matrix3d> S_arr;
    //  Python : i from -100 to 100
    for (int i = -100; i < 101; ++i) {
        double z_coord = i * 0.1;
        Vector3d obs_pt(z_coord, 0.4, 0.0);
        
        //  (eps  1e-3  Python)
        Matrix3d S = adaptive_integrate(obs_pt, P1, P2, P3, F_vec, lam, mu);
        S_arr.push_back(S);
        // 
        if (i % 20 == 0) {
            std::cout << "i = " << i << " | z = " << z_coord << "\n" 
                      << "Stress (0,0): " << S(0,0) << std::endl;
        }
    }
    save_stress_results("stress_results.txt", S_arr);

    std::cout << "Done." << std::endl;
    return 0;
}



Matrix3d adaptive_integrate_optimized(
    const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
    const Vector3d& F, double lam, double mu,
    gsl_integration_workspace* w_outer, 
    gsl_integration_workspace* w_inner) 
{
    double side1 = (p2 - p1).norm();
    double side2 = (p3 - p2).norm();
    double side3 = (p1 - p3).norm();
    double max_side = std::max({side1, side2, side3});
    double eps=1.0/20.0*max_side;
    Vector3d centroid = (p1 + p2 + p3) / 3.0;
    double dist = (obs - centroid).norm();

    // 1. ：（）
    if (dist > 5.0 * max_side) {
        return integrate_triangle_gauss(obs, p1, p2, p3, F, lam, mu);
    } 
    
    // 2. ： GSL （）
    Matrix3d total_sigma = Matrix3d::Zero();
    double nu = lam / (2.0 * (lam + mu));
    double jacobian = (p2 - p1).cross(p3 - p1).norm();

    IntegrationParams base_params;
    base_params.obs_pt = obs; base_params.p1 = p1; 
    base_params.v1 = p2 - p1; base_params.v2 = p3 - p1;
    base_params.F = F; base_params.nu = nu;
    base_params.eps=eps;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            base_params.comp_i = i;
            base_params.comp_j = j;

            OuterParams op = {&base_params, w_inner};
            gsl_function F_outer;
            F_outer.function = &outer_integrand;
            F_outer.params = &op;

            double res, err;
            //  w_outer
            gsl_integration_qags(&F_outer, 0, 1.0, 0, 1e-2, 1000, w_outer, &res, &err);
            total_sigma(i, j) = res * jacobian;
        }
    }
    return total_sigma;
}


void compute_adaptive_batch_core(
    const double* obs_ptr, size_t n_obs,
    const double* src_ptr, size_t n_src,
    const double* f_ptr, size_t n_f,
    double lam, double mu,
    double* out_ptr,
    bool use_parallel) // Added switch for parallel execution
{
    gsl_set_error_handler_off();

    // OpenMP parallel region with conditional execution based on use_parallel switch
    #pragma omp parallel if(use_parallel)
    {
        // Each thread must have its own GSL workspace to ensure thread safety
        gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(1000);
        gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000);

        // Collapse loops to improve parallel granularity and load balancing
        #pragma omp for schedule(dynamic) collapse(2)
        for (size_t i = 0; i < n_src; ++i) {
            size_t f_idx = (n_f == 1) ? 0 : i;
            Vector3d F = Map<const Vector3d>(&f_ptr[f_idx * 3]);

            for (size_t j = 0; j < n_obs; ++j) {
                Vector3d obs = Map<const Vector3d>(&obs_ptr[j * 3]);
                Vector3d p1  = Map<const Vector3d>(&src_ptr[i * 9 + 0]);
                Vector3d p2  = Map<const Vector3d>(&src_ptr[i * 9 + 3]);
                Vector3d p3  = Map<const Vector3d>(&src_ptr[i * 9 + 6]);

                Matrix3d sigma = adaptive_integrate_optimized(obs, p1, p2, p3, F, lam, mu, w_outer, w_inner);

                double* res_ptr = &out_ptr[(i * n_obs + j) * 6];
                res_ptr[0] = sigma(0, 0); // sxx
                res_ptr[1] = sigma(1, 1); // syy
                res_ptr[2] = sigma(2, 2); // szz
                res_ptr[3] = sigma(0, 1); // sxy
                res_ptr[4] = sigma(0, 2); // sxz
                res_ptr[5] = sigma(1, 2); // syz
            }
        }

        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
    }
}

py::array_t<double> py_compute_adaptive_stresses(
    py::array_t<double> obs_points, 
    py::array_t<double> source_tris, 
    py::array_t<double> F_list, 
    double lam, double mu,
    bool parallel = true) // Added optional parameter
{
    // Ensure input arrays are continuous
    auto obs_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obs_points);
    auto src_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(source_tris);
    auto f_c   = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(F_list);

    auto obs_info = obs_c.request();
    auto src_info = src_c.request();
    auto f_info   = f_c.request();

    size_t n_obs = obs_info.shape[0];
    size_t n_src = src_info.shape[0];
    size_t n_f   = (f_info.ndim == 1 && f_info.shape[0] == 3) ? 1 : f_info.shape[0];

    auto result = py::array_t<double>({ (ssize_t)n_src, (ssize_t)n_obs, (ssize_t)6 });
    auto res_info = result.request();

    {
        py::gil_scoped_release release;
        compute_adaptive_batch_core(
            (const double*)obs_info.ptr, n_obs,
            (const double*)src_info.ptr, n_src,
            (const double*)f_info.ptr, n_f,
            lam, mu,
            (double*)res_info.ptr,
            parallel // Passed switch
        );
    }

    return result;
}

void compute_displacement_batch_core(
    const double* obs_ptr, size_t n_obs,
    const double* src_ptr, size_t n_src,
    const double* f_ptr, size_t n_f,
    double lam, double mu,
    double* out_ptr,
    bool use_parallel) // 
{
    gsl_set_error_handler_off();
    
    //  if(use_parallel)  OpenMP 
    #pragma omp parallel if(use_parallel)
    {
        // ：Workspace 
        gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(1000);
        gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000);

        //  schedule(dynamic) 
        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t i = 0; i < n_src; ++i) 
        {
            size_t f_idx = (n_f == 1) ? 0 : i;
            const Vector3d F = Map<const Vector3d>(&f_ptr[f_idx * 3]);

            for (size_t j = 0; j < n_obs; ++j) {
                Vector3d obs = Map<const Vector3d>(&obs_ptr[j * 3]);
                Vector3d p1  = Map<const Vector3d>(&src_ptr[i * 9 + 0]);
                Vector3d p2  = Map<const Vector3d>(&src_ptr[i * 9 + 3]);
                Vector3d p3  = Map<const Vector3d>(&src_ptr[i * 9 + 6]);

                Vector3d disp = adaptive_integrate_disp(obs, p1, p2, p3, F, lam, mu, w_outer, w_inner);

                double* res_ptr = &out_ptr[(i * n_obs + j) * 3];
                res_ptr[0] = disp(0);
                res_ptr[1] = disp(1);
                res_ptr[2] = disp(2);
            }
        }
        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
    }
}


py::array_t<double> py_compute_adaptive_displacements(
    py::array_t<double> obs_points, 
    py::array_t<double> source_tris, 
    py::array_t<double> F_list, 
    double lam, double mu,
    bool parallel = true) //  True
{
    auto obs_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obs_points);
    auto src_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(source_tris);
    auto f_c   = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(F_list);

    auto obs_info = obs_c.request();
    auto src_info = src_c.request();
    auto f_info   = f_c.request();

    size_t n_obs = obs_info.shape[0];
    size_t n_src = src_info.shape[0];
    size_t n_f   = (f_info.ndim == 1 && f_info.shape[0] == 3) ? 1 : f_info.shape[0];

    auto result = py::array_t<double>({ (ssize_t)n_src, (ssize_t)n_obs, (ssize_t)3 });
    auto res_info = result.request();

    {
        py::gil_scoped_release release;
        compute_displacement_batch_core(
            (const double*)obs_info.ptr, n_obs,
            (const double*)src_info.ptr, n_src,
            (const double*)f_info.ptr, n_f,
            lam, mu,
            (double*)res_info.ptr,
            parallel // 
        );
    }

    return result;
}



PYBIND11_MODULE(kelvin_stress, m) {
    m.doc() = "PyQuake3D Adaptive Kelvin Displacement and Stress Library";

    m.def("compute_kelvin_stresses", &py_compute_adaptive_stresses,
          "Compute stresses using adaptive GSL integration (N x M x 6)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("F_list"),
          py::arg("lam"),
          py::arg("mu"),
          py::arg("parallel") = true);
    
    m.def("compute_kelvin_displacements", &py_compute_adaptive_displacements,
          "Compute displacements using adaptive GSL integration (N x M x 3)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("F_list"),
          py::arg("lam"),
          py::arg("mu"),
          py::arg("parallel") = true);

}




