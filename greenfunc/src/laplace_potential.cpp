#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> //  array_t

namespace py = pybind11;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 
struct LaplaceParams {
    double obs[3], p1[3], v1[3], v2[3], normal[3];
    double epsilon;
    double current_u;
    int kernel_type; // 0: , 1: 
};

/**
 *  1:  P*
 */
inline double get_laplace_kernel(const double obs[3], const double src[3], double eps) {
    double dx = obs[0] - src[0];
    double dy = obs[1] - src[1];
    double dz = obs[2] - src[2];
    double r2 = dx*dx + dy*dy + dz*dz;
    double eps2 = eps * eps;
    double R2 = r2 + eps2;
    double R = std::sqrt(R2);
    
    // : (2*r^2 + 5*eps^2) / (8 * PI * (r^2 + eps^2)^1.5)
    return (2.0 * r2 + 5.0 * eps2) / (8.0 * M_PI * R2 * R);
}


//  2:  dP/dn

inline double get_laplace_grad_kernel(const double obs[3], const double src[3], const double normal[3], double eps) {
    double r_vec[3] = {obs[0] - src[0], obs[1] - src[1], obs[2] - src[2]};
    double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
    double r_dot_n = r_vec[0]*normal[0] + r_vec[1]*normal[1] + r_vec[2]*normal[2];
    
    double eps2 = eps * eps;
    double R2 = r2 + eps2;
    
    // : -(2*r^2 + 11*eps^2) * (r·n) / (8 * PI * (r^2 + eps^2)^2.5)
    double numerator = -(2.0 * r2 + 11.0 * eps2) * r_dot_n;
    double denominator = 8.0 * M_PI * (R2 * R2 * std::sqrt(R2));
    
    return numerator / denominator;
}

// ---  ---
double laplace_inner_func(double v, void* p) {
    LaplaceParams* lp = (LaplaceParams*)p;
    double src[3];
    for(int i=0; i<3; ++i) 
        src[i] = lp->p1[i] + lp->current_u * lp->v1[i] + v * lp->v2[i];
    
    if (lp->kernel_type == 1) {
        return get_laplace_grad_kernel(lp->obs, src, lp->normal, lp->epsilon);
    }
    return get_laplace_kernel(lp->obs, src, lp->epsilon);
}

// (laplace_outer_func ， lp)
double laplace_outer_func(double u, void* p) {
    LaplaceParams* lp = (LaplaceParams*)p;
    lp->current_u = u;
    gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
    double res, err;
    gsl_function F_inner = {&laplace_inner_func, lp};
    gsl_integration_qags(&F_inner, 0, 1.0 - u, 0, 1e-6, 1000, w, &res, &err);
    gsl_integration_workspace_free(w);
    return res;
}

struct GaussPt {
    double l1, l2, l3, w;
};

// 7 ( Degree 5)
const std::vector<GaussPt> TRI_GAUSS_7 = {
    {0.33333333333333, 0.33333333333333, 0.33333333333333, 0.22500000000000},
    {0.79742698535309, 0.10128650732346, 0.10128650732346, 0.12593918054483},
    {0.10128650732346, 0.79742698535309, 0.10128650732346, 0.12593918054483},
    {0.10128650732346, 0.10128650732346, 0.79742698535309, 0.12593918054483},
    {0.05971587178977, 0.47014206410512, 0.47014206410512, 0.13239415278851},
    {0.47014206410512, 0.05971587178977, 0.47014206410512, 0.13239415278851},
    {0.47014206410512, 0.47014206410512, 0.05971587178977, 0.13239415278851}
};

/**
 * 
 */
double integrate_far_field(const double obs[3], const double v0[3], const double v1[3], const double v2[3], 
                           const double normal[3], double eps, int type) {
    // ... (Jacobian  area ) ...
    double e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    double e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    double cp[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
    double area = 0.5 * std::sqrt(cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2]);

    double integral = 0.0;
    for (const auto& pt : TRI_GAUSS_7) {
        double src[3];
        for (int i = 0; i < 3; ++i) src[i] = pt.l1 * v0[i] + pt.l2 * v1[i] + pt.l3 * v2[i];
        
        if (type == 1) integral += pt.w * get_laplace_grad_kernel(obs, src, normal, eps);
        else integral += pt.w * get_laplace_kernel(obs, src, eps);
    }
    return integral * area;
}

/**
 * 
 * mode: 0 , 1 
 */
std::vector<double> compute_all_laplace_potentials(
    const std::vector<std::vector<double>>& obs_points,
    const double P1[3], const double P2[3], const double P3[3],
    int mode = 0) 
{
    // 1. 
    double v1[3] = {P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]};
    double v2[3] = {P3[0]-P1[0], P3[1]-P1[1], P3[2]-P1[2]};
    double cp[3] = {v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]};
    double jacobian = std::sqrt(cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2]);
    double s1 = std::sqrt(std::pow(P2[0]-P1[0],2) + std::pow(P2[1]-P1[1],2) + std::pow(P2[2]-P1[2],2));
    double s2 = std::sqrt(std::pow(P3[0]-P2[0],2) + std::pow(P3[1]-P2[1],2) + std::pow(P3[2]-P2[2],2));
    double s3 = std::sqrt(std::pow(P1[0]-P3[0],2) + std::pow(P1[1]-P3[1],2) + std::pow(P1[2]-P3[2],2));
    
    double max_side = std::max(s1, std::max(s2, s3));
    double eps=1.0/20.0*max_side;
    double centroid[3] = {(P1[0]+P2[0]+P3[0])/3.0, (P1[1]+P2[1]+P3[1])/3.0, (P1[2]+P2[2]+P3[2])/3.0};
    //  ()
    double normal[3] = {cp[0]/jacobian, cp[1]/jacobian, cp[2]/jacobian};
    
    

    std::vector<double> results;
    results.reserve(obs_points.size());

    for (const auto& pt : obs_points) {
        double dist = std::sqrt(std::pow(pt[0]-centroid[0],2) + std::pow(pt[1]-centroid[1],2) + std::pow(pt[2]-centroid[2],2));
        
        if (dist > 5.0 * max_side) {
            results.push_back(integrate_far_field(pt.data(), P1, P2, P3, normal, eps, mode));
        } else {
            LaplaceParams lp;
            for(int i=0; i<3; ++i) { 
                lp.obs[i]=pt[i]; lp.p1[i]=P1[i]; lp.v1[i]=v1[i]; lp.v2[i]=v2[i]; lp.normal[i]=normal[i]; 
            }
            lp.epsilon = eps;
            lp.kernel_type = mode;

            gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
            double res, err;
            gsl_function F_outer = {&laplace_outer_func, &lp};
            gsl_integration_qags(&F_outer, 0, 1.0, 0, 1e-6, 1000, w, &res, &err);
            gsl_integration_workspace_free(w);
            results.push_back(res * jacobian);
        }
    }
    return results;
}


/**
 * 
 * param obs_points:  m*3 ()  3 ()
 * param source_points:  n*9 ()  9 ()
 * return: std::vector<std::vector<double>>  [n_triangles][m_points]
 */
std::vector<std::vector<double>> compute_potential_matrix(
    const std::vector<double>& obs_points,
    const std::vector<double>& source_points,
    int mode = 0) 
{
    // 1.  ( m*3)
    std::vector<std::vector<double>> formatted_obs;
    if (obs_points.size() == 3) {
        formatted_obs.push_back(obs_points);
    } else {
        for (size_t i = 0; i < obs_points.size(); i += 3) {
            formatted_obs.push_back({obs_points[i], obs_points[i+1], obs_points[i+2]});
        }
    }

    // 2.  ( n*9)
    struct Triangle { double p1[3], p2[3], p3[3]; };
    std::vector<Triangle> triangles;
    for (size_t i = 0; i < source_points.size(); i += 9) {
        Triangle t;
        for (int j = 0; j < 3; ++j) {
            t.p1[j] = source_points[i + j];
            t.p2[j] = source_points[i + 3 + j];
            t.p3[j] = source_points[i + 6 + j];
        }
        triangles.push_back(t);
    }

    size_t n = triangles.size();
    size_t m = formatted_obs.size();
    
    //  [n][m]
    std::vector<std::vector<double>> results_matrix(n, std::vector<double>(m));

    // 3. 
    for (size_t i = 0; i < n; ++i) {
        // 
        // 
        std::vector<double> row_results = compute_all_laplace_potentials(
            formatted_obs, 
            triangles[i].p1, 
            triangles[i].p2, 
            triangles[i].p3, 
            mode
        );
        
        results_matrix[i] = std::move(row_results);
    }

    return results_matrix;
}






void save_laplace_to_txt(const std::string& filename, const std::vector<double>& results) {
    std::ofstream outfile(filename);
    outfile << std::fixed << std::setprecision(10);
    outfile << "Index\tPotential\n";
    for (size_t i = 0; i < results.size(); ++i) {
        outfile << i << "\t" << results[i] << "\n";
    }
    outfile.close();
}




int main() {
    // 1.  GSL ，
    gsl_set_error_handler_off();

   //  (：)
    double P1[3] = {0.0, 0.0, 0.5};
    double P2[3] = {1.0, 0.0, 0.0};
    double P3[3] = {0.0, 1.0, 0.0};

    // 3.  epsilon ( r=0 )
    //  image_a8a777.png  P* = 1/(4*pi*r)， eps  1/(4*pi*sqrt(r^2 + eps^2))
    double eps = 1e-3;

    // 4. ：x=0.3, y=0.4, z  -300  300， 0.1
    std::vector<std::vector<double>> obs_points;
    double start_z = -30.0;
    double end_z = 30.0;
    double step = 0.1;
    int mode=1;
    int num_steps = static_cast<int>((end_z - start_z) / step);

    for (int i = 0; i <= num_steps; ++i) {
        double current_z = start_z + i * step;
        // ，
        if (current_z > end_z + 1e-9) break;
        obs_points.push_back({0.3, 0.4, current_z});
    }

    std::cout << "Generated " << obs_points.size() << " observation points." << std::endl;
    std::cout << "Computing Laplace Potentials (Scalar field)..." << std::endl;

    // 5. 
    //  5  Gauss  GSL 
    std::vector<double> results = compute_all_laplace_potentials(obs_points, P1, P2, P3, mode);

    // 6.  10 
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "\nPreview of first 10 results:\n";
    std::cout << "Index |      Z-Coord |    Potential (phi)\n";
    std::cout << "------------------------------------------\n";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(12) << obs_points[i][2] << " | " 
                  << std::setw(15) << results[i] << std::endl;
    }

    // 7.  (Tab )
    std::string filename = "laplace_potential_results1.txt";
    save_laplace_to_txt(filename, results);

    std::cout << "\nTask completed. Results saved to " << filename << std::endl;



    // 1.  (2)
    std::vector<double> obs = {
        0.5, 0.5, 1.0,  // 0
        0.5, 0.5, 5.0   // 1
    };

    // 2.  (1)
    std::vector<double> sources = {
        0.0, 0.0, 0.0, 
        1.0, 0.0, 0.0, 
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 
        2.0, 0.0, 0.0, 
        0.0, 1.0, 1.0
    };

    // 3. 
    try {
        auto potentials = compute_potential_matrix(obs, sources,mode);

        // 4. 
        for (size_t i = 0; i < potentials.size(); ++i) {
            for (size_t j = 0; j < potentials[i].size(); ++j) {
                std::cout << "Triangle " << i << " to Point " << j 
                          << " Potential: " << potentials[i][j] << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;

}



// std::vector<std::vector<double>> compute_potential_matrix(
//     const std::vector<double>& obs_points,
//     const std::vector<double>& source_points,
//     double eps_input, int mode);

// PYBIND11_MODULE(laplace_lib, m) {
//     m.doc() = "Laplace potential and gradient computation module";
    
//     m.def("compute_potential_matrix", &compute_potential_matrix,
//           "Compute a potential matrix (N x M)",
//           py::arg("obs_points"),
//           py::arg("source_points"),
//           py::arg("eps_input") = 1e-3,
//           py::arg("mode") = 0);
// }




// ---  ---
py::array_t<double> py_compute_potential_matrix(
    py::array_t<double> obs_points,    // : m*3  numpy 
    py::array_t<double> source_points, // : n*9  numpy  
    int mode) 
{
    gsl_set_error_handler_off();
    // 1. 
    auto buf_obs = obs_points.request();
    auto buf_src = source_points.request();

    // （）
    size_t m = (buf_obs.ndim == 1) ? buf_obs.shape[0] / 3 : buf_obs.shape[0];
    size_t n = (buf_src.ndim == 1) ? buf_src.shape[0] / 9 : buf_src.shape[0];

    // 
    double* ptr_obs = static_cast<double*>(buf_obs.ptr);
    double* ptr_src = static_cast<double*>(buf_src.ptr);

    // 2.  formatted_obs (，)
    // ，
    std::vector<std::vector<double>> formatted_obs(m, std::vector<double>(3));
    for (size_t j = 0; j < m; ++j) {
        formatted_obs[j] = { ptr_obs[j*3], ptr_obs[j*3+1], ptr_obs[j*3+2] };
    }

    // 3.  (N x M)，pybind11 
    auto result = py::array_t<double>({ (ssize_t)n, (ssize_t)m });
    auto buf_res = result.request();
    double* ptr_res = static_cast<double*>(buf_res.ptr);

    // 4. 
    
    for (size_t i = 0; i < n; ++i) {
        //  ptr_src  3 
        double* p1 = &ptr_src[i * 9];
        double* p2 = &ptr_src[i * 9 + 3];
        double* p3 = &ptr_src[i * 9 + 6];

        // 
        std::vector<double> row = compute_all_laplace_potentials(
            formatted_obs, p1, p2, p3, mode
        );

        //  numpy 
        for (size_t j = 0; j < m; ++j) {
            ptr_res[i * m + j] = row[j];
        }
    }

    return result;
}

PYBIND11_MODULE(laplace_lib, m) {
    m.doc() = "Laplace potential and gradient computation module";
    m.def("compute_pressure_potential", &py_compute_potential_matrix,
          "Compute a pressure potential matrix (N x M) from triangular sources",
          py::arg("obs_points"),
          py::arg("source_points"),
          py::arg("mode") = 0);
}

