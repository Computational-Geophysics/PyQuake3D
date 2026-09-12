#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
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

/**
 *  S_ij 
 *  3x3 
 */
Matrix3d calculate_Stresspotential(const Vector3d& r, double nu, double eps) {
    double r2 = r.squaredNorm();
    double R2 = r2 + eps * eps;
    double R = std::sqrt(R2);
    double R3 = R2 * R;
    
    double coeff = -1.0 / (8.0 * M_PI * (1.0 - nu) * R3);
    double common_term = R2 + nu * eps * eps;
    double om2nu = 1.0 - 2.0 * nu;

    //  Eigen  (outer product)  r_i * r_j
    Matrix3d S = common_term * Matrix3d::Identity() - om2nu * (r * r.transpose());
    
    return coeff * S;
}

/**
 *  dS_ij/dn 
 *  3x3 
 */
Matrix3d calculate_dStresspotential_dn(const Vector3d& r, const Vector3d& n, double nu, double eps) {
    double r2 = r.squaredNorm();
    double R2 = r2 + eps * eps;
    double R = std::sqrt(R2);
    double R3 = R2 * R;
    double R5 = R3 * R2;
    
    double r_dot_n = r.dot(n);
    double coeff = -1.0 / (8.0 * M_PI * (1.0 - nu));
    double om2nu = 1.0 - 2.0 * nu;

    // : - (r·n / R^5) * (R^2 + 3*nu*eps^2) * delta_ij
    Matrix3d term1 = -(r_dot_n / R5) * (R2 + 3.0 * nu * eps * eps) * Matrix3d::Identity();
    
    // : - (1-2nu / R^3) * (n_i*x_j + n_j*x_i)
    //  (n_i*x_j + n_j*x_i)  (n * r^T + r * n^T)
    Matrix3d term2 = -(om2nu / R3) * (n * r.transpose() + r * n.transpose());
    
    // : (3*(1-2nu)*(r·n) / R^5) * x_i*x_j
    Matrix3d term3 = (3.0 * om2nu * r_dot_n / R5) * (r * r.transpose());
    
    return coeff * (term1 + term2 + term3);
}


// --- GSL Integration Functions ---
struct IntegrationParams {
    Vector3d obs_pt;
    Vector3d p1, v1, v2,normal;
    Vector3d F;
    double lam, mu, nu,eps;
    int comp_i, comp_j;
    double u_val; // <--- Added this to track the outer integral state
    int kernel_type; // 0: , 1: 
};
// Struct helper for nested integration
struct OuterParams {
    IntegrationParams* base;
    gsl_integration_workspace* w_inner;
};
// Inner integral (v direction)
double inner_integrand(double v, void* params) {
    auto* p = static_cast<IntegrationParams*>(params);
    
    // P = p1 + u*v1 + v*v2
    Vector3d source_pt = p->p1 + (p->u_val * p->v1) + (v * p->v2);
    Vector3d r = p->obs_pt - source_pt;
    
    // Calculate stress at this specific point
    if (p->kernel_type == 0) 
    {
        Matrix3d sigma = calculate_Stresspotential(r, p->nu,p->eps);
        return sigma(p->comp_i, p->comp_j);
    }
    else
    {
        Matrix3d sigma_n = calculate_dStresspotential_dn(r,p->normal, p->nu,p->eps);
        return sigma_n(p->comp_i, p->comp_j);
    }
    
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

// --- Quadrature & Logic ---
Matrix3d integrate_triangle_gauss(const Vector3d& obs, const Vector3d& v1, const Vector3d& v2, const Vector3d& v3, 
                                 const Vector3d& normal, double lam, double mu,double eps,int mode) {
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
        if (mode == 0)
        {
            total_sigma += w[k] * calculate_Stresspotential(obs - qp, nu,eps);
        }
        else if(mode == 1)
        {
            total_sigma += w[k] * calculate_dStresspotential_dn(obs - qp,normal,nu,eps);
        }
        
    }

    return total_sigma * area;
}



Matrix3d adaptive_integrate(const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
                            double lam, double mu,int mode) {
    
    double side1 = (p2 - p1).norm();
    double side2 = (p3 - p2).norm();
    double side3 = (p1 - p3).norm();
    double max_side = std::max({side1, side2, side3});
    double eps=1.0/20.0*max_side;
    
    
    Vector3d centroid = (p1 + p2 + p3) / 3.0;
    double dist = (obs - centroid).norm();
    Vector3d v12 = p2 - p1;
    Vector3d v13 = p3 - p1;
    // 2.  (Cross Product)
    // ：，（）
    Vector3d normal = v12.cross(v13);
    double normal_norm = normal.norm();
    if (normal_norm > 0.0) {
        normal /= normal_norm;
    }
    
    if (dist > 5.0 * max_side) {
        return integrate_triangle_gauss(obs, p1, p2, p3, normal, lam, mu,eps,mode);
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
        base_params.normal = normal; base_params.nu = nu;
        base_params.eps=eps; base_params.kernel_type=mode;

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
    // 
    double mu = 20e8, nu = 0.3;
    double lam = 2 * mu * nu / (1 - 2 * nu);
    int mode=0;

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
        Matrix3d S = adaptive_integrate(obs_pt, P1, P2, P3, lam, mu,mode);
        S_arr.push_back(S);
        // 
        if (i % 20 == 0) {
            std::cout << "i = " << i << " | z = " << z_coord << "\n" 
                      << "Stress (0,0): " << S(0,0) << std::endl;
        }
    }
    save_stress_results("stressP_results.txt", S_arr);

    std::cout << "Done." << std::endl;
    return 0;

}



Matrix3d adaptive_integrate_optimized(
    const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
    double lam, double mu,int mode,
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
    Vector3d v12 = p2 - p1;
    Vector3d v13 = p3 - p1;
    Vector3d normal = v12.cross(v13);
    // 1. ：（）
    if (dist > 5.0 * max_side) {
        //return integrate_triangle_gauss(obs, p1, p2, p3, F, lam, mu);
        return integrate_triangle_gauss(obs, p1, p2, p3, normal, lam, mu,eps,mode);
    } 
    
    // 2. ： GSL （）
    Matrix3d total_sigma = Matrix3d::Zero();
    double nu = lam / (2.0 * (lam + mu));
    double jacobian = (p2 - p1).cross(p3 - p1).norm();

    IntegrationParams base_params;
    base_params.obs_pt = obs; base_params.p1 = p1; 
    base_params.v1 = p2 - p1; base_params.v2 = p3 - p1;
    base_params.normal = normal; base_params.nu = nu;
    base_params.eps=eps; base_params.kernel_type=mode;

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
            gsl_integration_qags(&F_outer, 0, 1.0, 0, 1e-4, 1000, w_outer, &res, &err);
            total_sigma(i, j) = res * jacobian;
        }
    }
    return total_sigma;
}



// ： N  M 
void compute_adaptive_batch_core(
    const double* obs_ptr, size_t n_obs,
    const double* src_ptr, size_t n_src,
    double lam, double mu,int mode,
    double* out_ptr) 
{
    gsl_set_error_handler_off();
    //  OpenMP 
    //#pragma omp parallel
    {
        // ，
        gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(1000);
        gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000);

        //#pragma omp for collapse(2)
        for (size_t i = 0; i < n_src; ++i) {
            for (size_t j = 0; j < n_obs; ++j) {
                // 1.  Eigen 
                Vector3d obs = Map<const Vector3d>(&obs_ptr[j * 3]);
                Vector3d p1  = Map<const Vector3d>(&src_ptr[i * 9 + 0]);
                Vector3d p2  = Map<const Vector3d>(&src_ptr[i * 9 + 3]);
                Vector3d p3  = Map<const Vector3d>(&src_ptr[i * 9 + 6]);
                //Vector3d F   = Map<const Vector3d>(&f_ptr[i * 3]);

                // 2.  ( adaptive_integrate  workspace)
                //  w_outer  w_inner
                //Matrix3d sigma = adaptive_integrate_optimized(obs, p1, p2, p3, F, lam, mu, w_outer, w_inner);
                //Matrix3d sigma = adaptive_integrate_optimized(obs, p1, p2, p3, lam, mu,mode, w_outer, w_inner);
                Matrix3d sigma = adaptive_integrate(obs,  p1, p2, p3, lam, mu,mode);
                // std::cout << "obs:\n" << obs << std::endl;
                // std::cout << "p1:\n" << p1 << std::endl;
                // std::cout << "p2:\n" << p2 << std::endl;
                // std::cout << "p3:\n" << p3 << std::endl;
                // std::cout << "Computed Sigma Matrix:\n" << sigma << std::endl;
                // 3.  (， 6 )
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
    double lam, double mu,int mode) 
{
    auto obs_info = obs_points.request();
    auto src_info = source_tris.request();

    size_t n_obs = obs_info.shape[0];
    size_t n_src = src_info.shape[0];

    //  (N, M, 6)
    auto result = py::array_t<double>({ (ssize_t)n_src, (ssize_t)n_obs, (ssize_t)6 });
    auto res_info = result.request();

    //  GIL ， C++ 
    py::gil_scoped_release release;

    compute_adaptive_batch_core(
        (const double*)obs_info.ptr, n_obs,
        (const double*)src_info.ptr, n_src,
        lam, mu,mode,
        (double*)res_info.ptr
    );

    return result;
}

PYBIND11_MODULE(kelvin_stress_potential, m) {
    m.doc() = "PyQuake3D Adaptive Kelvin Stress potential Library";

    m.def("compute_stresses_potential", &py_compute_adaptive_stresses,
          "Compute stresses using adaptive GSL integration (N x M x 6)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("lam"),
          py::arg("mu"),
          py::arg("mode")
        );
}

