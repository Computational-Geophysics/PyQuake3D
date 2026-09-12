
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <iomanip>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // 必须包含这个才能使用 array_t

namespace py = pybind11;
using namespace Eigen;


// --- 1. 结构体定义 (确保 OuterParams 在使用前定义) ---
struct IntegrationParams {
    Vector3d obs_pt;
    Vector3d p1, v1, v2;
    Vector3d delta_u, n;
    double lam, mu,nu, eps;
    int comp_l;      // 统一使用 comp_l
    int comp_i, comp_j; // 明确指定矩阵的行和列
    double u_val; 
};

// struct TractionIntegrationParams {
//     Vector3d obs_pt;
//     Vector3d p1, v1, v2;
//     Vector3d delta_u; // 位错量
//     Vector3d n_element; // 积分面单元法向
//     double nu, mu, eps;
//     double u_val;
//     // 移除 comp_i/comp_j，因为我们直接计算完整的 Matrix3d 牵引力贡献
// };


struct OuterParams {
    IntegrationParams* base;
    gsl_integration_workspace* w_inner;
};

// struct OuterParams {
//     void* base; // 通用指针，支持 IntegrationParams 或 StressIntegrationParams
//     gsl_integration_workspace* w_inner;
// };


Matrix3d rotate_matrix(
    const Vector3d& P1,
    const Vector3d& P2,
    const Vector3d& P3)
{
    Vector3d Vnorm = (P2 - P1).cross(P3 - P1);
    Vnorm.normalize();

    const Vector3d eX(1.0, 0.0, 0.0);
    const Vector3d eZ(0.0, 0.0, 1.0);

    Vector3d Vstrike = eZ.cross(Vnorm);

    if (Vstrike.norm() < 1e-12)
    {
        Vstrike = eX * Vnorm.z();
    }

    Vstrike.normalize();

    Vector3d Vdip = Vnorm.cross(Vstrike);

    Matrix3d At;
    At.col(0) = Vstrike;
    At.col(1) = Vdip;
    At.col(2) = Vnorm;

    return At;
}

// Helper: Get specific component of G_{lp,q}
inline double get_G_derivative_lpq(int l, int p, int q, const Vector3d& diff, double mu, double lam, double eps) {
    double r2 = diff.squaredNorm();
    double R2 = r2 + eps * eps;
    double R = std::sqrt(R2);
    double R3 = R2 * R;
    double R5 = R2 * R3;

    double C1 = 1.0 / (8.0 * M_PI * mu);
    double C2 = (lam + mu) / (8.0 * M_PI * mu * (lam + 2.0 * mu));

    double term1 = (l == p) ? -((2.0 * r2 + 5.0 * eps * eps) * diff(q)) / R5 : 0.0;
    double delta_lp_xq = (l == p ? 1.0 : 0.0) * diff(q);
    double delta_lq_xp = (l == q ? 1.0 : 0.0) * diff(p);
    double delta_pq_xl = (p == q ? 1.0 : 0.0) * diff(l);

    double bracket = (delta_lp_xq + delta_lq_xp + delta_pq_xl) / R3;
    double sub_term = (3.0 * diff(l) * diff(p) * diff(q)) / R5;

    return C1 * term1 + C2 * (bracket - sub_term);
}

// 核心内核
double get_dislocation_displacement_component(const Vector3d& diff, const Vector3d& delta_u, const Vector3d& n, 
                                             int comp_l, double mu, double lam, double eps) {
    double dot_dn = delta_u.dot(n);
    double div_G = 0;
    for(int i = 0; i < 3; ++i) div_G += get_G_derivative_lpq(comp_l, i, i, diff, mu, lam, eps);
    
    double term_dilation = lam * dot_dn * div_G;
    double term_shear = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            term_shear += mu * delta_u(i) * n(j) * (get_G_derivative_lpq(comp_l, i, j, diff, mu, lam, eps) + 
                                                   get_G_derivative_lpq(comp_l, j, i, diff, mu, lam, eps));
        }
    }
    return -(term_dilation + term_shear);
}

// 积分函数
double dislocation_inner_integrand(double v, void* params) {
    auto* p = static_cast<IntegrationParams*>(params);
    Vector3d qp = p->p1 + (p->u_val * p->v1) + (v * p->v2);
    Vector3d diff = p->obs_pt - qp;
    return get_dislocation_displacement_component(diff, p->delta_u, p->n, p->comp_l, p->mu, p->lam, p->eps);
}

double dislocation_outer_integrand(double u, void* params) {
    auto* op = reinterpret_cast<OuterParams*>(params);
   
    auto* base = reinterpret_cast<IntegrationParams*>(op->base);
    base->u_val = u; 
    
    gsl_function F_inner;
    F_inner.function = &dislocation_inner_integrand;
    F_inner.params = base; // 传回转换后的指针

    double result, error;
    gsl_integration_qag(&F_inner, 0, 1.0 - u, 0, 1e-2, 1000,GSL_INTEG_GAUSS15, op->w_inner, &result, &error);
    return result;
}


Vector3d integrate_dislocation_gauss(const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, 
                                      const Vector3d& delta_u, const Vector3d& n, double lam, double mu, double eps) {
    static const std::vector<double> w = {0.225, 0.125939, 0.125939, 0.125939, 0.132394, 0.132394, 0.132394};
    static const std::vector<Vector3d> b = {
        {1.0/3, 1.0/3, 1.0/3}, {0.797427, 0.101287, 0.101287}, {0.101287, 0.797427, 0.101287},
        {0.101287, 0.101287, 0.797427}, {0.059716, 0.470142, 0.470142}, {0.470142, 0.059716, 0.470142},
        {0.470142, 0.470142, 0.059716}
    };

    Vector3d total_disp = Vector3d::Zero();
    double area = 0.5 * (p2 - p1).cross(p3 - p1).norm();

    for (int k = 0; k < 7; ++k) {
        Vector3d qp = b[k](0) * p1 + b[k](1) * p2 + b[k](2) * p3;
        Vector3d diff = obs - qp;
        for (int l = 0; l < 3; ++l) {
            total_disp(l) += w[k] * get_dislocation_displacement_component(diff, delta_u, n, l, mu, lam, eps);
        }
    }
    return total_disp * area;
}


Vector3d adaptive_integrate_disp(const Vector3d& obs, const Vector3d& p1, const Vector3d& p2, const Vector3d& p3,
                                 const Vector3d& slip, double lam, double mu,
                                 gsl_integration_workspace* w_outer, 
                                 gsl_integration_workspace* w_inner,double eps_ratio=0.05) 
{
    // Geometry calculations
    double side1 = (p2 - p1).norm();
    double side2 = (p3 - p2).norm();
    double side3 = (p1 - p3).norm();
    double max_side = std::max({side1, side2, side3});
    double v1[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    double v2[3] = {p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2]};
    double cp[3] = {v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]};
    double jacobian = std::sqrt(cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2]);
    Vector3d normal = Vector3d(cp[0], cp[1], cp[2]) / jacobian;
    //std::cout << "Normal vector: " << normal << std::endl;
    // Regularization parameter proportional to triangle size
    double eps = eps_ratio * max_side;
    if(eps_ratio>1.0)
    {
        eps=eps_ratio;
    }
    Vector3d centroid = (p1 + p2 + p3) / 3.0;
    double dist = (obs - centroid).norm();

    Matrix3d At = rotate_matrix(p1, p2, p3);
    Vector3d delta_u = At * slip;

    // 1. Far-field: Use 7-point Gauss Quadrature for speed
    if (dist > 5.0 * max_side) {
        return integrate_dislocation_gauss(obs, p1, p2, p3, delta_u, normal, lam, mu, eps);
    } 
    
    // 2. Near-field: Use GSL Nested Adaptive Integration
    Vector3d total_disp = Vector3d::Zero();

    IntegrationParams base_params;
    base_params.obs_pt = obs; 
    base_params.p1 = p1; 
    base_params.v1 = p2 - p1; 
    base_params.v2 = p3 - p1;
    base_params.delta_u = delta_u;
    base_params.n = normal;
    base_params.lam = lam; 
    base_params.mu = mu;
    base_params.eps = eps;

    // Perform integration for each displacement component (l = 0, 1, 2)
    for (int l = 0; l < 3; ++l) {
        base_params.comp_l = l;
        OuterParams op = {&base_params, w_inner};
        
        gsl_function F_outer;
        F_outer.function = &dislocation_outer_integrand;
        F_outer.params = &op;

        double res, err;
        // Integrating over u from 0 to 1, with inner integration domain (v) from 0 to 1-u
        //gsl_integration_qags(&F_outer, 0, 1.0, 0, 1e-4, 1000, w_outer, &res, &err);
        gsl_integration_qag(&F_outer, 0, 1.0, 0, 1e-4, 1000,GSL_INTEG_GAUSS15, w_outer, &res, &err);
        total_disp(l) = res * jacobian;
    }
    return total_disp;
}










// void get_regularized_m_derivative(
//     const Vector3d& r_vec,
//     double eps,
//     double out[3][3][3][3])
// {
//     double r_sq  = r_vec.squaredNorm();
//     double eps_sq = eps * eps;

//     double R_sq = r_sq + eps_sq;
//     double R = std::sqrt(R_sq);

//     double R5 = R_sq * R_sq * R;      // (r²+eps²)^(5/2)
//     double R7 = R5 * R_sq;            // (r²+eps²)^(7/2)

//     double coeff = -1.0 / (4.0 * M_PI);

//     auto delta = [](int a, int b) {
//         return (a == b) ? 1.0 : 0.0;
//     };

//     double A = (r_sq + 2.5 * eps_sq) / R5;
//     double B = 3.0 * (r_sq + 3.5 * eps_sq) / R7;

//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             for (int k = 0; k < 3; ++k) {
//                 for (int q = 0; q < 3; ++q) {

//                     out[i][j][k][q] =
//                         coeff * delta(i, k) *
//                         (
//                             delta(j, q) * A
//                             - r_vec(j) * r_vec(q) * B
//                         );
//                 }
//             }
//         }
//     }
// }




// void get_stress_derivative(const Vector3d& r_vec, double nu, double eps, double out[3][3][3][3]) {
//     double r_sq = r_vec.squaredNorm();
//     double eps_sq = eps * eps;
//     double R_sq = r_sq + eps_sq;
//     double R = std::sqrt(R_sq);
//     double R2 = R_sq;
//     double R5 = R2 * R2 * R;
//     double R7 = R5 * R2;

//     double A = -1.0 / (8.0 * M_PI * (1.0 - nu));
//     double nu_factor = nu / (1.0 - 2.0 * nu);

//     auto delta = [](int i, int j) { return (i == j) ? 1.0 : 0.0; };

//     // double mout[3][3][3][3];
//     // get_regularized_m_derivative(r_vec, eps, mout); 

//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             for (int k = 0; k < 3; ++k) {
//                 for (int q = 0; q < 3; ++q) {
//                     // 公式第一项
//                     double term1 = (1.0 - 2.0 * nu) * (
//                         (delta(i, k)*delta(j, q) + delta(j, k)*delta(i, q) - delta(i, j)*delta(k, q)) * (r_sq + 1.5 * eps_sq) +
//                         (delta(i, k)*r_vec(j) + delta(j, k)*r_vec(i) - delta(i, j)*r_vec(k)) * (r_vec(q) * (2.0*R_sq - 5.0*(r_sq + 1.5*eps_sq)) / R_sq)
//                     );

//                     // 公式第二项
//                     double term2 = 3.0 * (
//                         delta(i, q)*r_vec(j)*r_vec(k) + delta(j, q)*r_vec(i)*r_vec(k) + delta(k, q)*r_vec(i)*r_vec(j) - 
//                         5.0 * r_vec(i)*r_vec(j)*r_vec(k)*r_vec(q) / R_sq
//                     );

//                     // 公式第三项
//                     double term3 = -1.5 * eps_sq * (
//                         (delta(i, k)*delta(j, q) + delta(j, k)*delta(i, q) + delta(i, j)*delta(k, q)*nu_factor) -
//                         5.0 * r_vec(q) * (delta(i, k)*r_vec(j) + delta(j, k)*r_vec(i) + delta(i, j)*r_vec(k)*nu_factor) / R_sq
//                     );

//                     out[i][j][k][q] = A * (term1 / R5 + term2 / R5 + term3 / R5);
//                 }
//             }
//         }
//     }
    

// }



void get_stress_derivative(
    const Vector3d& r_vec,
    double nu,
    double eps,
    double out[3][3][3][3])
{
    const double r_sq   = r_vec.squaredNorm();
    const double eps_sq = eps * eps;
    const double R_sq   = r_sq + eps_sq;

    // 当 eps = 0 且 r = 0 时，经典核存在奇异性
    if (R_sq <= 0.0) {
        throw std::runtime_error(
            "get_stress_derivative: singular point with r = 0 and eps = 0.");
    }

    const double R  = std::sqrt(R_sq);
    const double R5 = R_sq * R_sq * R;

    const double one_minus_2nu = 1.0 - 2.0 * nu;
    const double prefactor =
        -1.0 / (8.0 * M_PI * (1.0 - nu) * R5);

    // 修正后应力核中的两个系数
    const double coef_A =
        one_minus_2nu * r_sq
        + (4.0 - 5.0 * nu) * eps_sq;

    const double coef_B =
        one_minus_2nu * r_sq
        + (1.0 - 5.0 * nu) * eps_sq;

    auto delta = [](int a, int b) -> double {
        return (a == b) ? 1.0 : 0.0;
    };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int q = 0; q < 3; ++q) {

                    const double xi = r_vec(i);
                    const double xj = r_vec(j);
                    const double xk = r_vec(k);
                    const double xq = r_vec(q);

                    /*
                     * S_ijk = delta_ik*x_j + delta_jk*x_i
                     * D_ijk = delta_ij*x_k
                     */
                    const double S =
                        delta(i, k) * xj
                        + delta(j, k) * xi;

                    const double D =
                        delta(i, j) * xk;

                    /*
                     * N_ijk =
                     * A*(delta_ik*x_j + delta_jk*x_i)
                     * - B*delta_ij*x_k
                     * + 3*x_i*x_j*x_k
                     */
                    const double N =
                        coef_A * S
                        - coef_B * D
                        + 3.0 * xi * xj * xk;

                    /*
                     * dS_ijk/dx_q =
                     * delta_ik*delta_jq + delta_jk*delta_iq
                     */
                    const double dS =
                        delta(i, k) * delta(j, q)
                        + delta(j, k) * delta(i, q);

                    /*
                     * dD_ijk/dx_q = delta_ij*delta_kq
                     */
                    const double dD =
                        delta(i, j) * delta(k, q);

                    /*
                     * d(x_i*x_j*x_k)/dx_q
                     */
                    const double dCubic =
                        delta(i, q) * xj * xk
                        + delta(j, q) * xi * xk
                        + delta(k, q) * xi * xj;

                    /*
                     * dA/dx_q = dB/dx_q
                     *          = 2*(1 - 2*nu)*x_q
                     */
                    const double dCoef =
                        2.0 * one_minus_2nu * xq;

                    /*
                     * dN_ijk/dx_q
                     */
                    const double dN =
                        dCoef * (S - D)
                        + coef_A * dS
                        - coef_B * dD
                        + 3.0 * dCubic;

                    /*
                     * Sigma_ijk,q =
                     * -1/[8*pi*(1-nu)*R^5]
                     * [dN_ijk/dx_q - 5*x_q*N_ijk/R^2]
                     */
                    out[i][j][k][q] =
                        prefactor *
                        (dN - 5.0 * xq * N / R_sq);
                }
            }
        }
    }
}


Matrix3d compute_stress_kernel(const Vector3d& obs, const Vector3d& xi, const Vector3d& n_xi, 
                               const Vector3d& delta_u, double mu, double nu, double eps) {
    Vector3d r_vec = obs - xi;
    double dSigma[3][3][3][3];
    get_stress_derivative(r_vec, nu, eps, dSigma);

    double lambda = 2.0 * mu * nu / (1.0 - 2.0 * nu);
    Matrix3d stress = Matrix3d::Zero();
    
    // 预计算 lambda 相关部分，减少循环内的冗余计算
    // 假设 n_xi 和 delta_u 的点积在循环外固定
    double delta_u_dot_n = delta_u.dot(n_xi);
    // std::cout<<delta_u_dot_n<<std::endl;
    
    double r2 = r_vec.squaredNorm();
    double denom = std::pow(r2 + eps * eps, 3.5);
    double fai = 15.0 * std::pow(eps*1.0, 4) /(8.0*M_PI  * denom);
    
    // double eps1=eps*1.0;
    // double r2 = r_vec.squaredNorm();
    // double R2 = r2 + eps1 * eps1;
    // double denom = R2 * R2 * R2 * std::sqrt(R2); 
    // double fai =
    //     15.0 * std::pow(eps1, 4)
    //     / (8.0 * M_PI * denom);

    // 本征应变
    Matrix3d eps_plastic =
        0.5 *
        (delta_u * n_xi.transpose()
         + n_xi * delta_u.transpose())
        * fai;

    // 本征应力 C:eps*
    //Matrix3d eigenstress_term1=eps_plastic.trace();
    // Matrix3d eigenstress =
    //     lambda * eps_plastic.trace() * Matrix3d::Identity()
    //     + 2.0*mu * eps_plastic;

    

    for (int k = 0; k < 3; ++k) {
        for (int l = k; l < 3; ++l) { // 修改点：l 从 k 开始，只计算上三角
            
            // 优化：提取出 dSigma[k][l][0][0] + dSigma[k][l][1][1] + dSigma[k][l][2][2]
            double dSigma_trace = dSigma[k][l][0][0] + dSigma[k][l][1][1] + dSigma[k][l][2][2];
            double term_lambda =  delta_u_dot_n * dSigma_trace;

            double term_mu = 0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    term_mu +=  delta_u(i) * n_xi(j) * (dSigma[k][l][i][j] + dSigma[k][l][j][i]);
                }
            }
            
            double val = -lambda*(term_lambda+eps_plastic.trace()) - (term_mu+2.0*eps_plastic(k,l))*mu;
            
            //double val = -(term_lambda + term_mu);
            stress(k, l) = val;
            //std::cout<<val<<std::endl;
            
            
            // 利用对称性填充下三角
            if (k != l) {
                stress(l, k) = val;
            }
        }
    }
    return stress;
}






// ============================================================
// Duffy-transformed near-field triangle integration
//
// For a subtriangle (Q, A, B), use
//   x(s,t) = Q + s * [(1-t)(A-Q) + t(B-Q)],
//   0 <= s <= 1, 0 <= t <= 1.
//
// The surface Jacobian is
//   J = s * |(A-Q) x (B-Q)|.
//
// Choosing Q as the closest point on the source triangle to the
// observation point moves the sharpest part of the regularized
// kernel to s = 0. The factor s in the Jacobian makes the
// transformed integrand much better behaved there.
// ============================================================
struct DuffyParams {
    Vector3d obs;
    Vector3d Q, A, B;
    Vector3d delta_u;
    Vector3d normal;
    double mu, nu, eps;
    int comp_i, comp_j;
    double s;
    gsl_integration_workspace* w_inner;
};


double fixed_gauss_legendre_integral(
    gsl_function* function,
    double lower,
    double upper,
    size_t order)
{
    gsl_integration_glfixed_table* table =
        gsl_integration_glfixed_table_alloc(order);
    if (table == nullptr) {
        throw std::runtime_error(
            "Unable to allocate the Gauss-Legendre integration table."
        );
    }

    const double result =
        gsl_integration_glfixed(function, lower, upper, table);
    gsl_integration_glfixed_table_free(table);

    if (!std::isfinite(result)) {
        throw std::runtime_error(
            "Gauss-Legendre fallback produced a non-finite Duffy integral."
        );
    }
    return result;
}


// Inner Duffy integral over t in [0,1]
double duffy_inner_integrand(double t, void* params) {
    auto* p = static_cast<DuffyParams*>(params);

    // At exactly s=0, the area Jacobian is zero. Returning zero
    // avoids evaluating the strongest kernel point unnecessarily.
    if (p->s <= 1e-15) {
        return 0.0;
    }

    const Vector3d QA = p->A - p->Q;
    const Vector3d QB = p->B - p->Q;

    const Vector3d direction = (1.0 - t) * QA + t * QB;
    const Vector3d qp = p->Q + p->s * direction;

    const double jac = p->s * QA.cross(QB).norm();

    const Matrix3d T = compute_stress_kernel(
        p->obs, qp, p->normal, p->delta_u,
        p->mu, p->nu, p->eps
    );

    return T(p->comp_i, p->comp_j) * jac;
}


// Outer Duffy integral over s in [0,1]
double duffy_outer_integrand(double s, void* params) {
    auto* p = static_cast<DuffyParams*>(params);
    p->s = s;

    gsl_function F_inner;
    F_inner.function = &duffy_inner_integrand;
    F_inner.params = p;

    double result = 0.0;
    double error = 0.0;

    // The Duffy transform already regularizes the numerical shape
    // of the near-field integrand, so there is no need for an
    // extremely aggressive tolerance here.
    constexpr double epsabs_inner = 1e-9;
    constexpr double epsrel_inner = 1e-6;
    constexpr size_t limit = 1000;

    int status = gsl_integration_qag(
        &F_inner,
        0.0, 1.0,
        epsabs_inner,
        epsrel_inner,
        limit,
        GSL_INTEG_GAUSS31,
        p->w_inner,
        &result,
        &error
    );

    // Retry any failed adaptive integration with a higher-order rule.
    if (status != GSL_SUCCESS) {
        status = gsl_integration_qag(
            &F_inner,
            0.0, 1.0,
            epsabs_inner,
            epsrel_inner,
            limit,
            GSL_INTEG_GAUSS61,
            p->w_inner,
            &result,
            &error
        );
    }

    // A fixed high-order rule is bounded and cannot exhaust an adaptive
    // subdivision limit. Use it instead of retaining a failed QAG result.
    if (status != GSL_SUCCESS) {
        result = fixed_gauss_legendre_integral(
            &F_inner, 0.0, 1.0, 96
        );
    }

    return result;
}


// Expand the narrow regularized layer near s=0 without changing eps or the
// value of the integral. With s=u^3 and ds=3*u^2 du,
// integral F(s) ds equals integral F(u^3)*3*u^2 du on [0,1].
double duffy_outer_transformed_integrand(double u, void* params) {
    if (u <= 0.0) {
        return 0.0;
    }

    const double u2 = u * u;
    return duffy_outer_integrand(u2 * u, params) * (3.0 * u2);
}


// Integrate one Duffy subtriangle (Q,A,B) for one stress component.
double integrate_duffy_subtriangle(
    const Vector3d& obs,
    const Vector3d& Q,
    const Vector3d& A,
    const Vector3d& B,
    const Vector3d& delta_u,
    const Vector3d& normal,
    double mu,
    double nu,
    double eps,
    int comp_i,
    int comp_j,
    gsl_integration_workspace* w_outer,
    gsl_integration_workspace* w_inner)
{
    const Vector3d QA = A - Q;
    const Vector3d QB = B - Q;

    const double area_jac = QA.cross(QB).norm();

    // Q can lie on an edge or at a vertex. In that case one or
    // two subtriangles are degenerate and should simply contribute 0.
    const double scale2 = std::max(QA.squaredNorm(), QB.squaredNorm());
    const double degenerate_tol = 1e-14 * std::max(1.0, scale2);
    if (area_jac <= degenerate_tol) {
        return 0.0;
    }

    DuffyParams p;
    p.obs = obs;
    p.Q = Q;
    p.A = A;
    p.B = B;
    p.delta_u = delta_u;
    p.normal = normal;
    p.mu = mu;
    p.nu = nu;
    p.eps = eps;
    p.comp_i = comp_i;
    p.comp_j = comp_j;
    p.s = 0.0;
    p.w_inner = w_inner;

    gsl_function F_outer;
    F_outer.function = &duffy_outer_transformed_integrand;
    F_outer.params = &p;

    // When obs == Q, the regularized peak is located exactly at the Duffy
    // endpoint. Bounded Gauss-Legendre quadrature after the cubic transform
    // is both faster and more reliable than exhausting adaptive subdivisions.
    const double edge_scale = std::max(QA.norm(), QB.norm());
    const double on_surface_tolerance =
        1e-12 * std::max(1.0, edge_scale);
    if ((obs - Q).norm() <= on_surface_tolerance) {
        return fixed_gauss_legendre_integral(
            &F_outer, 0.0, 1.0, 64
        );
    }

    double result = 0.0;
    double error = 0.0;

    constexpr double epsabs_outer = 1e-7;
    constexpr double epsrel_outer = 1e-5;
    constexpr size_t limit = 5000;

    int status = gsl_integration_qag(
        &F_outer,
        0.0, 1.0,
        epsabs_outer,
        epsrel_outer,
        limit,
        GSL_INTEG_GAUSS31,
        w_outer,
        &result,
        &error
    );

    // QAGS is better suited to difficult endpoint behavior.
    if (status != GSL_SUCCESS) {
        status = gsl_integration_qags(
            &F_outer,
            0.0, 1.0,
            epsabs_outer,
            epsrel_outer,
            limit,
            w_outer,
            &result,
            &error
        );
    }

    // Fall back to bounded high-order quadrature rather than returning a
    // failed adaptive result or emitting repeated convergence warnings.
    if (status != GSL_SUCCESS) {
        result = fixed_gauss_legendre_integral(
            &F_outer, 0.0, 1.0, 96
        );
    }

    return result;
}


// Integrate all stress components together for an observation whose closest
// point Q lies on the source triangle. This avoids repeating the complete
// stress-kernel calculation once for every tensor component.
Matrix3d integrate_duffy_subtriangle_on_surface(
    const Vector3d& obs,
    const Vector3d& Q,
    const Vector3d& A,
    const Vector3d& B,
    const Vector3d& delta_u,
    const Vector3d& normal,
    double mu,
    double nu,
    double eps)
{
    const Vector3d QA = A - Q;
    const Vector3d QB = B - Q;
    const double area_jac = QA.cross(QB).norm();
    const double scale2 = std::max(QA.squaredNorm(), QB.squaredNorm());
    const double degenerate_tol = 1e-14 * std::max(1.0, scale2);
    if (area_jac <= degenerate_tol) {
        return Matrix3d::Zero();
    }

    constexpr size_t radial_order = 64;
    constexpr size_t angular_order = 48;
    gsl_integration_glfixed_table* radial_table =
        gsl_integration_glfixed_table_alloc(radial_order);
    gsl_integration_glfixed_table* angular_table =
        gsl_integration_glfixed_table_alloc(angular_order);
    if (radial_table == nullptr || angular_table == nullptr) {
        if (radial_table != nullptr) {
            gsl_integration_glfixed_table_free(radial_table);
        }
        if (angular_table != nullptr) {
            gsl_integration_glfixed_table_free(angular_table);
        }
        throw std::runtime_error(
            "Unable to allocate on-surface Duffy quadrature tables."
        );
    }

    Matrix3d result = Matrix3d::Zero();
    for (size_t iu = 0; iu < radial_order; ++iu) {
        double u = 0.0;
        double wu = 0.0;
        gsl_integration_glfixed_point(
            0.0, 1.0, iu, &u, &wu, radial_table
        );
        const double u2 = u * u;
        const double s = u2 * u;
        const double transformed_jac = 3.0 * u2 * s * area_jac;

        for (size_t it = 0; it < angular_order; ++it) {
            double t = 0.0;
            double wt = 0.0;
            gsl_integration_glfixed_point(
                0.0, 1.0, it, &t, &wt, angular_table
            );
            const Vector3d direction = (1.0 - t) * QA + t * QB;
            const Vector3d qp = Q + s * direction;
            result += wu * wt * transformed_jac * compute_stress_kernel(
                obs, qp, normal, delta_u, mu, nu, eps
            );
        }
    }

    gsl_integration_glfixed_table_free(radial_table);
    gsl_integration_glfixed_table_free(angular_table);
    if (!result.allFinite()) {
        throw std::runtime_error(
            "On-surface Duffy quadrature produced non-finite stress."
        );
    }
    return result;
}


Matrix3d compute_traction_gauss(const Vector3d& obs, const Vector3d& p1, 
                                const Vector3d& p2, const Vector3d& p3, 
                                const Vector3d& delta_u, const Vector3d& n, 
                                double nu, double mu, double eps) {
    
    // 7点 Gauss-Triangle 积分权重与重心坐标
    static const double w[] = {0.225, 0.125939, 0.125939, 0.125939, 0.132394, 0.132394, 0.132394};
    static const double b[7][3] = {
        {1.0/3, 1.0/3, 1.0/3}, {0.797427, 0.101287, 0.101287}, {0.101287, 0.797427, 0.101287},
        {0.101287, 0.101287, 0.797427}, {0.059716, 0.470142, 0.470142}, {0.470142, 0.059716, 0.470142},
        {0.470142, 0.470142, 0.059716}
    };

    Matrix3d total_tensor = Matrix3d::Zero();
    // 计算三角形面积用于面积分映射
    double area = 0.5 * (p2 - p1).cross(p3 - p1).norm();

    for (int k = 0; k < 7; ++k) {
        // 重心坐标映射到全局笛卡尔空间 qp
        Vector3d qp = b[k][0] * p1 + b[k][1] * p2 + b[k][2] * p3;
        
        // 获取核张量 T_kli (3x3x3)
        // 注意：计算逻辑应为 T_ij = T_kli * n_k * delta_u_l
        Matrix3d T = compute_stress_kernel(obs, qp, n,delta_u, mu, nu, eps);

        
        
        // 累加：Matrix3d(T_ij) 与加权面积项相乘
        // 确保在此处计算的是完整的张量分量
        total_tensor += w[k] * T; 
    }
    
    return total_tensor * area;
}


double pointTriangleDistance(
    const Eigen::Vector3d& P,
    const Eigen::Vector3d& A,
    const Eigen::Vector3d& B,
    const Eigen::Vector3d& C,
    Eigen::Vector3d& closestPoint)
{
    const Eigen::Vector3d AB = B - A;
    const Eigen::Vector3d AC = C - A;
    const Eigen::Vector3d AP = P - A;

    const double d1 = AB.dot(AP);
    const double d2 = AC.dot(AP);

    // 顶点 A 区域
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        closestPoint = A;
        return (P - closestPoint).norm();
    }

    const Eigen::Vector3d BP = P - B;

    const double d3 = AB.dot(BP);
    const double d4 = AC.dot(BP);

    // 顶点 B 区域
    if (d3 >= 0.0 && d4 <= d3)
    {
        closestPoint = B;
        return (P - closestPoint).norm();
    }

    const double vc = d1*d4 - d3*d2;

    // 边 AB 区域
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double v = d1 / (d1 - d3);

        closestPoint = A + v*AB;

        return (P - closestPoint).norm();
    }

    const Eigen::Vector3d CP = P - C;

    const double d5 = AB.dot(CP);
    const double d6 = AC.dot(CP);

    // 顶点 C 区域
    if (d6 >= 0.0 && d5 <= d6)
    {
        closestPoint = C;
        return (P - closestPoint).norm();
    }

    const double vb = d5*d2 - d1*d6;

    // 边 AC 区域
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double w = d2 / (d2 - d6);

        closestPoint = A + w*AC;

        return (P - closestPoint).norm();
    }

    const double va = d3*d6 - d5*d4;

    // 边 BC 区域
    if (va <= 0.0 &&
        (d4 - d3) >= 0.0 &&
        (d5 - d6) >= 0.0)
    {
        const double w =
            (d4 - d3) /
            ((d4 - d3) + (d5 - d6));

        closestPoint = B + w*(C - B);

        return (P - closestPoint).norm();
    }

    // 面内部区域
    const double inverseSum = 1.0 / (va + vb + vc);
    const double v = vb * inverseSum;
    const double w = vc * inverseSum;

    closestPoint = A + v*AB + w*AC;

    return (P - closestPoint).norm();
}


/**
 * 牵引力张量积分函数
 * 自动根据距离 obs 与单元的相对位置选择高斯积分或自适应积分
 */
Matrix3d adaptive_integrate_traction(
    const Vector3d& obs,
    const Vector3d& p1,
    const Vector3d& p2,
    const Vector3d& p3,
    const Vector3d& slip,
    double nu,
    double mu,
    gsl_integration_workspace* w_outer,
    gsl_integration_workspace* w_inner,
    double eps_ratio = 0.05)
{
    // --------------------------------------------------------
    // 1. Triangle geometry
    // --------------------------------------------------------
    const Vector3d v1 = p2 - p1;
    const Vector3d v2 = p3 - p1;
    const Vector3d cross_prod = v1.cross(v2);
    const double jacobian = cross_prod.norm();

    if (jacobian <= 1e-15) {
        throw std::runtime_error(
            "adaptive_integrate_traction: degenerate source triangle."
        );
    }

    const Vector3d normal = cross_prod / jacobian;

    const double side1 = (p2 - p1).norm();
    const double side2 = (p3 - p2).norm();
    const double side3 = (p1 - p3).norm();
    const double max_side = std::max({side1, side2, side3});

    // --------------------------------------------------------
    // 2. Regularization length
    // eps_ratio < 1 : interpreted as fraction of max side
    // eps_ratio >=1 : interpreted as an absolute length
    // --------------------------------------------------------
    double eps = eps_ratio * max_side;
    if (eps_ratio >= 1.0) {
        eps = eps_ratio;
    }

    if (!(eps > 0.0)) {
        throw std::runtime_error(
            "adaptive_integrate_traction: eps must be positive."
        );
    }

    // --------------------------------------------------------
    // 3. Rotate local slip into the global coordinate system
    // --------------------------------------------------------
    const Matrix3d At = rotate_matrix(p1, p2, p3);
    const Vector3d delta_u = At * slip;

    // --------------------------------------------------------
    // 4. Use the true point-to-triangle distance, not the
    //    distance to the centroid.
    //
    //    Q is the closest point on the triangle to obs.
    //    If obs projects inside the triangle, Q is that projection.
    // --------------------------------------------------------
    Vector3d Q;
    const double dist_to_triangle =
        pointTriangleDistance(obs, p1, p2, p3, Q);

    const double on_surface_tolerance =
        1e-12 * std::max(1.0, max_side);
    if (dist_to_triangle <= on_surface_tolerance) {
        return
            integrate_duffy_subtriangle_on_surface(
                obs, Q, p1, p2, delta_u, normal, mu, nu, eps
            ) +
            integrate_duffy_subtriangle_on_surface(
                obs, Q, p2, p3, delta_u, normal, mu, nu, eps
            ) +
            integrate_duffy_subtriangle_on_surface(
                obs, Q, p3, p1, delta_u, normal, mu, nu, eps
            );
    }

    // --------------------------------------------------------
    // 5. Far field: fast 7-point triangle quadrature
    // --------------------------------------------------------
    if (dist_to_triangle > 5.0 * max_side) {
        return compute_traction_gauss(
            obs, p1, p2, p3,
            delta_u, normal,
            nu, mu, eps
        );
    }

    // --------------------------------------------------------
    // 6. Near field: split the original triangle around Q
    //    and integrate each piece using a Duffy transform.
    //
    //       T1 = (Q,p1,p2)
    //       T2 = (Q,p2,p3)
    //       T3 = (Q,p3,p1)
    //
    //    If Q lies on an edge/vertex, degenerate pieces are
    //    detected and skipped automatically.
    // --------------------------------------------------------
    Matrix3d result = Matrix3d::Zero();

    static constexpr int components[6][2] = {
        {0, 0}, {1, 1}, {2, 2},
        {0, 1}, {0, 2}, {1, 2}
    };

    for (const auto& comp : components) {
        const int i = comp[0];
        const int j = comp[1];

        double value = 0.0;

        value += integrate_duffy_subtriangle(
            obs, Q, p1, p2,
            delta_u, normal,
            mu, nu, eps,
            i, j,
            w_outer, w_inner
        );

        value += integrate_duffy_subtriangle(
            obs, Q, p2, p3,
            delta_u, normal,
            mu, nu, eps,
            i, j,
            w_outer, w_inner
        );

        value += integrate_duffy_subtriangle(
            obs, Q, p3, p1,
            delta_u, normal,
            mu, nu, eps,
            i, j,
            w_outer, w_inner
        );

        // IMPORTANT:
        // Do NOT multiply by the old triangle jacobian here.
        // The physical area Jacobian is already included in the
        // Duffy integrand as
        //     s * |(A-Q) x (B-Q)|.
        result(i, j) = value;

        if (i != j) {
            result(j, i) = value;
        }
    }

    return result;
}


void save_displacement_results(const std::string& filename, const std::vector<Vector3d>& D_list) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // 写入表头 (Tab 分隔)
    outFile << "Index\tDx\tDy\tDz" << std::endl;

    // 设置浮点数精度
    outFile << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < D_list.size(); ++i) {
        const Vector3d& D = D_list[i];
        outFile << i << "\t"
                << D(0) << "\t"  // Dx
                << D(1) << "\t"  // Dy
                << D(2)          // Dz
                << std::endl;
    }

    outFile.close();
    std::cout << "Displacement results saved to " << filename << std::endl;
}


void save_stress_results(const std::string& filename, const std::vector<Matrix3d>& S_list) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // 写入表头 (Tab 分隔)
    outFile << "Index\tSxx\tSxy\tSxz\tSyy\tSyz\tSzz" << std::endl;

    // 设置浮点数精度 (对应你给出的 10 位小数左右)
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
    // 1. 定义物理参数
    double mu = 30e9; // 剪切模量
    double lam = 30e9;       
    double nu=lam/(2.0*(lam+mu));
    
    // 2. 定义源三角形顶点 (以原点为中心的三角形)
    Vector3d P1(1.0, 0.0, 0.0);
    Vector3d P2(0.0, 1.0, 0.0);
    Vector3d P3(0.0, 0.0, 0.0);
    
    // 3. 定义位错矢量 (delta_u) 和 法向量 (n)
    Vector3d delta_u(1.0, 1.0, 0.0); // Z方向位错 0.1m
    
    
    // 4. 初始化 GSL 工作空间 (在主函数中预分配，避免在循环中重复申请/释放)
    gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(5000);
    gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(5000);
    
    std::vector<Vector3d> disp_results;
    std::vector<Matrix3d> stress_results;

    Vector3d obs_pt(0.3, 0.2, 50.1);
    Vector3d disp= adaptive_integrate_disp(obs_pt, P1, P2, P3, delta_u, lam, mu, w_outer, w_inner);
    Matrix3d stress = adaptive_integrate_traction(obs_pt, P1, P2, P3, delta_u, nu, mu, w_outer, w_inner);
    //Matrix3d stres0 = compute_stress_kernel(obs_pt, P1, souce_normal, delta_u, nu, mu,eps=1e-3);

    std::cout << disp << std::endl;
    std::cout << stress << std::endl;
    //std::cout << stres0 << std::endl;

    // 5. 循环观测点 (例如：在 Z 轴上方扫描)
    for (int i = 0; i <= 200; ++i) {
        double x_val = -1+0.01*i;
        Vector3d obs_pt(x_val, 0.3, 0);
        
        // 调用自适应积分
        Vector3d disp = adaptive_integrate_disp(obs_pt, P1, P2, P3, delta_u, lam, mu, w_outer, w_inner);
        Matrix3d stress = adaptive_integrate_traction(obs_pt, P1, P2, P3, delta_u, nu, mu, w_outer, w_inner);
        stress_results.push_back(stress);
        
        disp_results.push_back(disp);
        std::cout << "obs_x=" << x_val << " | Displacement: " << disp.transpose() << std::endl;
    }
    save_displacement_results("dislocation_displacement_results.txt", disp_results);
    save_stress_results("stress.txt", stress_results);
    // 6. 释放空间
    gsl_integration_workspace_free(w_outer);
    gsl_integration_workspace_free(w_inner);
    
    // std::cout << "计算完成。" << std::endl;
    return 0;
}




void compute_displacement_batch_core(
    const double* const obs_ptr, size_t n_obs,
    const double* const src_ptr, size_t n_src,
    const double* const du_ptr, size_t n_du, 
    double lam, double mu,
    double* const res_ptr,double eps_ratio,
    bool use_parallel) // Added switch for parallel execution
{
    gsl_set_error_handler_off();

    // OpenMP parallel region with conditional execution based on use_parallel switch
    #pragma omp parallel if(use_parallel)
    {
        // Each thread must have its own GSL workspace to ensure thread safety
        gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(5000);
        gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(5000);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n_src; ++i) {
            // Re-use logic: share slip vector if only one is provided
            size_t du_idx = (n_du == 1) ? 0 : i;
            
            // Eigen mapping: zero-copy access to data pointers
            const Vector3d delta_u = Map<const Vector3d>(&du_ptr[du_idx * 3]);

            for (size_t j = 0; j < n_obs; ++j) {
                Vector3d obs(&obs_ptr[j * 3]);
                Vector3d p1(&src_ptr[i * 9 + 0]);
                Vector3d p2(&src_ptr[i * 9 + 3]);
                Vector3d p3(&src_ptr[i * 9 + 6]);

                Vector3d disp = adaptive_integrate_disp(
                    obs, p1, p2, p3, delta_u, lam, mu, 
                    w_outer, w_inner,eps_ratio
                );

                // Write results with offset (N_src x N_obs x 3)
                size_t offset = (i * n_obs + j) * 3;
                res_ptr[offset + 0] = disp(0);
                res_ptr[offset + 1] = disp(1);
                res_ptr[offset + 2] = disp(2);
            }
        }

        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
    }
}

void compute_displacement_batch_core_eps(
    const double* const obs_ptr,
    size_t n_obs,
    const double* const src_ptr,
    size_t n_src,
    const double* const du_ptr,
    size_t n_du,
    double lam,
    double mu,
    double* const res_ptr,
    const double* const eps_ratio_ptr,   // 长度为 n_src
    bool use_parallel)
{
    gsl_set_error_handler_off();

    #pragma omp parallel if(use_parallel)
    {
        gsl_integration_workspace* w_outer =
            gsl_integration_workspace_alloc(5000);

        gsl_integration_workspace* w_inner =
            gsl_integration_workspace_alloc(5000);

        #pragma omp for schedule(dynamic) collapse(2)
        for (size_t i = 0; i < n_src; ++i) {
            for (size_t j = 0; j < n_obs; ++j) {

                const size_t du_idx = (n_du == 1) ? 0 : i;

                const Vector3d delta_u =
                    Map<const Vector3d>(du_ptr + du_idx * 3);

                const Vector3d obs =
                    Map<const Vector3d>(obs_ptr + j * 3);

                const Vector3d p1 =
                    Map<const Vector3d>(src_ptr + i * 9 + 0);

                const Vector3d p2 =
                    Map<const Vector3d>(src_ptr + i * 9 + 3);

                const Vector3d p3 =
                    Map<const Vector3d>(src_ptr + i * 9 + 6);

                // 第 i 个源对应的正则化参数
                const double eps_ratio_i = eps_ratio_ptr[i];

                const Vector3d disp = adaptive_integrate_disp(
                    obs,
                    p1,
                    p2,
                    p3,
                    delta_u,
                    lam,
                    mu,
                    w_outer,
                    w_inner,
                    eps_ratio_i
                );

                const size_t offset = (i * n_obs + j) * 3;

                res_ptr[offset + 0] = disp(0);
                res_ptr[offset + 1] = disp(1);
                res_ptr[offset + 2] = disp(2);
            }
        }

        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
    }
}




py::array_t<double> py_compute_adaptive_displacements(
    py::array_t<double> obs_points, 
    py::array_t<double> source_tris, 
    py::array_t<double> delta_u_list, 
    double lam, double mu,double eps_ratio=0.05,
    bool parallel = true) // Set default to True
{
    // Ensure input arrays are contiguous and in memory
    auto obs_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obs_points);
    auto src_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(source_tris);
    auto du_c  = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(delta_u_list);

    auto obs_info = obs_c.request();
    auto src_info = src_c.request();
    auto du_info  = du_c.request();

    size_t n_obs = obs_info.shape[0];
    size_t n_src = src_info.shape[0];
    size_t n_du   = (du_info.ndim == 1 && du_info.shape[0] == 3) ? 1 : du_info.shape[0];

    // Pre-allocate result array
    std::vector<ssize_t> shape = { (ssize_t)n_src, (ssize_t)n_obs, 3 };
    auto result = py::array_t<double>(shape);
    py::buffer_info res_info = result.request();

    // Release GIL to allow multi-threaded C++ execution
    {
        py::gil_scoped_release release;
        compute_displacement_batch_core(
            (const double*)obs_info.ptr, n_obs,
            (const double*)src_info.ptr, n_src,
            (const double*)du_info.ptr, n_du,
            lam, mu,
            (double*)res_info.ptr,
            eps_ratio,parallel
        );
    }

    return result;
}


py::array_t<double> py_compute_adaptive_displacements_eps(
    py::array_t<double> obs_points,
    py::array_t<double> source_tris,
    py::array_t<double> delta_u_list,
    double lam,
    double mu,
    py::array_t<double> eps_ratio_list,   // 长度为 n_src
    bool parallel = true)
{
    auto obs_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            obs_points
        );

    auto src_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            source_tris
        );

    auto du_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            delta_u_list
        );

    auto eps_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            eps_ratio_list
        );

    if (!obs_c || !src_c || !du_c || !eps_c) {
        throw std::runtime_error(
            "Failed to convert input arrays to contiguous double arrays."
        );
    }

    const py::buffer_info obs_info = obs_c.request();
    const py::buffer_info src_info = src_c.request();
    const py::buffer_info du_info  = du_c.request();
    const py::buffer_info eps_info = eps_c.request();

    // obs_points: (n_obs, 3)
    if (obs_info.ndim != 2 || obs_info.shape[1] != 3) {
        throw py::value_error(
            "obs_points must have shape (n_obs, 3)."
        );
    }

    // source_tris: (n_src, 9) 或 (n_src, 3, 3)
    if (!(
        (src_info.ndim == 2 && src_info.shape[1] == 9) ||
        (src_info.ndim == 3 &&
         src_info.shape[1] == 3 &&
         src_info.shape[2] == 3)
    )) {
        throw py::value_error(
            "source_tris must have shape (n_src, 9) "
            "or (n_src, 3, 3)."
        );
    }

    const size_t n_obs =
        static_cast<size_t>(obs_info.shape[0]);

    const size_t n_src =
        static_cast<size_t>(src_info.shape[0]);

    // delta_u_list:
    // (3,), (1,3), 或 (n_src,3)
    size_t n_du = 0;

    if (du_info.ndim == 1) {
        if (du_info.shape[0] != 3) {
            throw py::value_error(
                "One-dimensional delta_u_list must have shape (3,)."
            );
        }

        n_du = 1;
    }
    else if (du_info.ndim == 2) {
        if (du_info.shape[1] != 3) {
            throw py::value_error(
                "Two-dimensional delta_u_list must have shape "
                "(1, 3) or (n_src, 3)."
            );
        }

        n_du = static_cast<size_t>(du_info.shape[0]);

        if (n_du != 1 && n_du != n_src) {
            throw py::value_error(
                "delta_u_list must have shape (3,), (1, 3), "
                "or (n_src, 3)."
            );
        }
    }
    else {
        throw py::value_error(
            "delta_u_list must have shape (3,), (1, 3), "
            "or (n_src, 3)."
        );
    }

    // eps_ratio_list 必须为 (n_src,)
    if (eps_info.ndim != 1) {
        throw py::value_error(
            "eps_ratio_list must be a one-dimensional array."
        );
    }

    if (static_cast<size_t>(eps_info.shape[0]) != n_src) {
        throw py::value_error(
            "eps_ratio_list length must equal n_src."
        );
    }

    const double* eps_ptr =
        static_cast<const double*>(eps_info.ptr);

    // 预分配输出：(n_src, n_obs, 3)
    const std::vector<ssize_t> shape = {
        static_cast<ssize_t>(n_src),
        static_cast<ssize_t>(n_obs),
        3
    };

    py::array_t<double> result(shape);
    const py::buffer_info res_info = result.request();

    {
        py::gil_scoped_release release;

        compute_displacement_batch_core_eps(
            static_cast<const double*>(obs_info.ptr),
            n_obs,
            static_cast<const double*>(src_info.ptr),
            n_src,
            static_cast<const double*>(du_info.ptr),
            n_du,
            lam,
            mu,
            static_cast<double*>(res_info.ptr),
            eps_ptr,
            parallel
        );
    }

    return result;
}


void compute_traction_batch_core(
    const double* const obs_ptr, size_t n_obs,
    const double* const src_ptr, size_t n_src,
    const double* const du_ptr, size_t n_du,
    double nu, double mu,
    double* const res_ptr,double eps_ratio,
    bool use_parallel) // 增加开关参数
{
    gsl_set_error_handler_off();

    // if(use_parallel) 只有当该值为 true 时才会并行化，否则串行执行
    #pragma omp parallel if(use_parallel)
    {
        // GSL workspace 在所有线程中分配（如果串行，则仅在主线程分配一个）
        gsl_integration_workspace* w_outer = gsl_integration_workspace_alloc(5000);
        gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(5000);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n_src; ++i) {
            size_t du_idx = (n_du == 1) ? 0 : i;
            const Vector3d delta_u = Map<const Vector3d>(&du_ptr[du_idx * 3]);

            for (size_t j = 0; j < n_obs; ++j) {
                Vector3d obs(&obs_ptr[j * 3]);
                Vector3d p1(&src_ptr[i * 9 + 0]);
                Vector3d p2(&src_ptr[i * 9 + 3]);
                Vector3d p3(&src_ptr[i * 9 + 6]);

                Matrix3d result = adaptive_integrate_traction(
                    obs, p1, p2, p3, delta_u, 
                    nu, mu, w_outer, w_inner,eps_ratio
                );

                size_t offset = (i * n_obs + j) * 6;
                res_ptr[offset + 0] = result(0, 0); 
                res_ptr[offset + 1] = result(1, 1);
                res_ptr[offset + 2] = result(2, 2);
                res_ptr[offset + 3] = result(0, 1);
                res_ptr[offset + 4] = result(0, 2);
                res_ptr[offset + 5] = result(1, 2);
            }
        }
        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
    }
}


py::array_t<double> py_adaptive_integrate_traction(
    py::array_t<double> obs_points, 
    py::array_t<double> source_tris, 
    py::array_t<double> delta_u_list, 
    double nu, double mu,double eps_ratio,
    bool parallel) // 新增并行控制参数
{
    // 强制转换为连续内存布局 (防止非连续切片导致的数据读取错误)
    auto obs_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obs_points);
    auto src_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(source_tris);
    auto du_c  = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(delta_u_list);

    auto obs_info = obs_c.request();
    auto src_info = src_c.request();
    auto du_info  = du_c.request();

    size_t n_obs = obs_info.shape[0];
    size_t n_src = src_info.shape[0];
    
    // 逻辑：如果 du 是 1x3 向量则全局复用；否则对应每个 src 单元
    size_t n_du = (du_info.ndim == 1) ? 1 : du_info.shape[0];

    // 预分配结果数组: N_src x N_obs x 6
    std::vector<ssize_t> shape_res = { (ssize_t)n_src, (ssize_t)n_obs, 6 };
    auto result = py::array_t<double>(shape_res);
    py::buffer_info res_info = result.request();

    // 在计算密集的循环前释放 Python 全局解释器锁 (GIL)
    {
        py::gil_scoped_release release;
        compute_traction_batch_core(
            (const double*)obs_info.ptr, n_obs,
            (const double*)src_info.ptr, n_src,
            (const double*)du_info.ptr, n_du,
            nu, mu,
            (double*)res_info.ptr,
            eps_ratio,parallel // 传入并行控制参数
        );
    }

    return result;
}


void compute_traction_batch_core_eps(
    const double* const obs_ptr,
    size_t n_obs,
    const double* const src_ptr,
    size_t n_src,
    const double* const du_ptr,
    size_t n_du,
    double nu,
    double mu,
    double* const res_ptr,
    const double* const eps_ratio_ptr,   // 长度为 n_src
    bool use_parallel)
{
    gsl_set_error_handler_off();

    #pragma omp parallel if(use_parallel)
    {
        // 每个 OpenMP 线程分别分配 GSL workspace，避免线程冲突
        gsl_integration_workspace* w_outer =
            gsl_integration_workspace_alloc(5000);

        gsl_integration_workspace* w_inner =
            gsl_integration_workspace_alloc(5000);

        #pragma omp for schedule(dynamic) collapse(2)
        for (size_t i = 0; i < n_src; ++i) {
            for (size_t j = 0; j < n_obs; ++j) {

                // delta_u：
                // n_du == 1 时所有源共用，否则第 i 个源使用第 i 个 delta_u
                const size_t du_idx = (n_du == 1) ? 0 : i;

                const Vector3d delta_u =
                    Map<const Vector3d>(du_ptr + du_idx * 3);

                const Vector3d obs =
                    Map<const Vector3d>(obs_ptr + j * 3);

                const Vector3d p1 =
                    Map<const Vector3d>(src_ptr + i * 9 + 0);

                const Vector3d p2 =
                    Map<const Vector3d>(src_ptr + i * 9 + 3);

                const Vector3d p3 =
                    Map<const Vector3d>(src_ptr + i * 9 + 6);

                // 第 i 个源对应的 eps_ratio
                const double eps_ratio_i = eps_ratio_ptr[i];

                const Matrix3d result = adaptive_integrate_traction(
                    obs,
                    p1,
                    p2,
                    p3,
                    delta_u,
                    nu,
                    mu,
                    w_outer,
                    w_inner,
                    eps_ratio_i
                );

                const size_t offset = (i * n_obs + j) * 6;

                res_ptr[offset + 0] = result(0, 0);
                res_ptr[offset + 1] = result(1, 1);
                res_ptr[offset + 2] = result(2, 2);
                res_ptr[offset + 3] = result(0, 1);
                res_ptr[offset + 4] = result(0, 2);
                res_ptr[offset + 5] = result(1, 2);
            }
        }

        gsl_integration_workspace_free(w_outer);
        gsl_integration_workspace_free(w_inner);
    }
}


py::array_t<double> py_adaptive_integrate_traction_eps(
    py::array_t<double> obs_points,
    py::array_t<double> source_tris,
    py::array_t<double> delta_u_list,
    double nu,
    double mu,
    py::array_t<double> eps_ratio_list,   // 长度为 n_src
    bool parallel)
{
    // 转换为 double 类型、C 连续内存
    auto obs_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            obs_points
        );

    auto src_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            source_tris
        );

    auto du_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            delta_u_list
        );

    auto eps_c =
        py::array_t<double,
                    py::array::c_style | py::array::forcecast>::ensure(
            eps_ratio_list
        );

    if (!obs_c || !src_c || !du_c || !eps_c) {
        throw std::runtime_error(
            "Failed to convert input arrays to contiguous double arrays."
        );
    }

    const py::buffer_info obs_info = obs_c.request();
    const py::buffer_info src_info = src_c.request();
    const py::buffer_info du_info  = du_c.request();
    const py::buffer_info eps_info = eps_c.request();

    // 检查 obs_points: (n_obs, 3)
    if (obs_info.ndim != 2 || obs_info.shape[1] != 3) {
        throw py::value_error(
            "obs_points must have shape (n_obs, 3)."
        );
    }

    // 检查 source_tris:
    // 支持 (n_src, 9) 或 (n_src, 3, 3)
    if (!(
        (src_info.ndim == 2 && src_info.shape[1] == 9) ||
        (src_info.ndim == 3 &&
         src_info.shape[1] == 3 &&
         src_info.shape[2] == 3)
    )) {
        throw py::value_error(
            "source_tris must have shape (n_src, 9) "
            "or (n_src, 3, 3)."
        );
    }

    const size_t n_obs =
        static_cast<size_t>(obs_info.shape[0]);

    const size_t n_src =
        static_cast<size_t>(src_info.shape[0]);

    // 检查 delta_u_list:
    // 可以是 (3,) 或 (1, 3)，表示所有源共用；
    // 也可以是 (n_src, 3)，表示每个源使用不同的 delta_u。
    size_t n_du = 0;

    if (du_info.ndim == 1) {
        if (du_info.shape[0] != 3) {
            throw py::value_error(
                "A one-dimensional delta_u_list must have shape (3,)."
            );
        }

        n_du = 1;
    }
    else if (du_info.ndim == 2) {
        if (du_info.shape[1] != 3) {
            throw py::value_error(
                "A two-dimensional delta_u_list must have shape "
                "(1, 3) or (n_src, 3)."
            );
        }

        n_du = static_cast<size_t>(du_info.shape[0]);

        if (n_du != 1 && n_du != n_src) {
            throw py::value_error(
                "delta_u_list must have shape (3,), (1, 3), "
                "or (n_src, 3)."
            );
        }
    }
    else {
        throw py::value_error(
            "delta_u_list must have shape (3,), (1, 3), "
            "or (n_src, 3)."
        );
    }

    // 检查 eps_ratio_list 必须是长度为 n_src 的一维数组
    if (eps_info.ndim != 1) {
        throw py::value_error(
            "eps_ratio_list must be a one-dimensional array "
            "with shape (n_src,)."
        );
    }

    if (static_cast<size_t>(eps_info.shape[0]) != n_src) {
        throw py::value_error(
            "The length of eps_ratio_list must equal n_src."
        );
    }

    // 可选：检查 eps_ratio 是否有效
    const double* eps_ptr =
        static_cast<const double*>(eps_info.ptr);

    for (size_t i = 0; i < n_src; ++i) {
        if (!std::isfinite(eps_ptr[i])) {
            throw py::value_error(
                "eps_ratio_list contains NaN or infinity."
            );
        }

        if (eps_ptr[i] <= 0.0) {
            throw py::value_error(
                "All values in eps_ratio_list must be greater than zero."
            );
        }
    }

    // 输出数组：(n_src, n_obs, 6)
    const std::vector<ssize_t> shape_res = {
        static_cast<ssize_t>(n_src),
        static_cast<ssize_t>(n_obs),
        6
    };

    py::array_t<double> result(shape_res);
    const py::buffer_info res_info = result.request();

    {
        py::gil_scoped_release release;

        compute_traction_batch_core_eps(
            static_cast<const double*>(obs_info.ptr),
            n_obs,
            static_cast<const double*>(src_info.ptr),
            n_src,
            static_cast<const double*>(du_info.ptr),
            n_du,
            nu,
            mu,
            static_cast<double*>(res_info.ptr),
            eps_ptr,
            parallel
        );
    }

    return result;
}

// 绑定模块
PYBIND11_MODULE(dislocation_lib, m) {
    m.def("compute_dislocation_stresses", &py_adaptive_integrate_traction,
          "Adaptive dislocation stresses caused by dislocations (returns N_src x N_obs x 6)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("delta_u_list"),
          py::arg("nu"),
          py::arg("mu"),
          py::arg("eps_ratio")=0.05,
          py::arg("parallel") = true);
    
    
    m.def("compute_dislocation_stresses_eps", &py_adaptive_integrate_traction_eps,
          "Adaptive dislocation stresses caused by dislocations (returns N_src x N_obs x 6)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("delta_u_list"),
          py::arg("nu"),
          py::arg("mu"),
          py::arg("eps_ratio"),
          py::arg("parallel") = true);
    
    m.def("compute_dislocation_displacements", &py_compute_adaptive_displacements,
          "Adaptive Compute displacements caused by dislocations (N x M x 3)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("delta_u_list"),
          py::arg("lam"),
          py::arg("mu"),
          py::arg("eps_ratio")=0.05,
          py::arg("parallel") = true);
    m.def("compute_dislocation_displacements_eps", &py_compute_adaptive_displacements_eps,
          "Adaptive Compute displacements caused by dislocations (N x M x 3)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("delta_u_list"),
          py::arg("lam"),
          py::arg("mu"),
          py::arg("eps_ratio"),
          py::arg("parallel") = true);
}
