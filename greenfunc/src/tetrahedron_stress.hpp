// This code is modified from source code by Barbot, Sylvain. 
//"Deformation of a Half‐Space from Anelastic Strain Confined in a Tetrahedral Volume." Bulletin of the Seismological Society of America (2018), doi: 10.1785/0120180058..

#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace tetra {

struct Vec3 {
    double x{}, y{}, z{};
    double& operator[](std::size_t i) { return i == 0 ? x : (i == 1 ? y : z); }
    double operator[](std::size_t i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

struct Eigenstrain {
    double e11{}, e12{}, e13{}, e22{}, e23{}, e33{};
};

struct Stress {
    double s11{}, s12{}, s13{}, s22{}, s23{}, s33{};
};

struct Tetrahedron { Vec3 A, B, C, D; };

// MATLAB-style interface. x1, x2 and x3 must have identical lengths.
// A, B, C and D are {north, east, depth}. Output vectors are resized.
void computeStressTetrahedronGauss(
    const std::vector<double>& x1, const std::vector<double>& x2,
    const std::vector<double>& x3, const std::array<double,3>& A,
    const std::array<double,3>& B, const std::array<double,3>& C,
    const std::array<double,3>& D, double e11, double e12, double e13,
    double e22, double e23, double e33, double G, double nu,
    std::vector<double>& s11, std::vector<double>& s12,
    std::vector<double>& s13, std::vector<double>& s22,
    std::vector<double>& s23, std::vector<double>& s33, int N = 15);

void computeStressTetrahedronTanhSinh(
    const std::vector<double>& x1, const std::vector<double>& x2,
    const std::vector<double>& x3, const std::array<double,3>& A,
    const std::array<double,3>& B, const std::array<double,3>& C,
    const std::array<double,3>& D, double e11, double e12, double e13,
    double e22, double e23, double e33, double G, double nu,
    std::vector<double>& s11, std::vector<double>& s12,
    std::vector<double>& s13, std::vector<double>& s22,
    std::vector<double>& s23, std::vector<double>& s33,
    double precision = 0.001, double bound = 3.5);

void computeStressTetrahedronMixedQuad(
    const std::vector<double>& x1, const std::vector<double>& x2,
    const std::vector<double>& x3, const std::array<double,3>& A,
    const std::array<double,3>& B, const std::array<double,3>& C,
    const std::array<double,3>& D, double e11, double e12, double e13,
    double e22, double e23, double e33, double G, double nu,
    std::vector<double>& s11, std::vector<double>& s12,
    std::vector<double>& s13, std::vector<double>& s22,
    std::vector<double>& s23, std::vector<double>& s33,
    int N = 7, double precision = 0.01, double bound = 3.0);

Stress computeStressTetrahedronGauss(
    const Vec3& observation, const Tetrahedron& source,
    const Eigenstrain& eigenstrain, double shear_modulus,
    double poisson_ratio, int order = 15);

Stress computeStressTetrahedronTanhSinh(
    const Vec3& observation, const Tetrahedron& source,
    const Eigenstrain& eigenstrain, double shear_modulus,
    double poisson_ratio, double precision = 0.001, double bound = 3.5);

Stress computeStressTetrahedronMixedQuad(
    const Vec3& observation, const Tetrahedron& source,
    const Eigenstrain& eigenstrain, double shear_modulus,
    double poisson_ratio, int gauss_order = 7,
    double tanh_precision = 0.01, double tanh_bound = 3.0);

std::vector<Stress> computeStressTetrahedronMixedQuad(
    const std::vector<Vec3>& observations, const Tetrahedron& source,
    const Eigenstrain& eigenstrain, double shear_modulus,
    double poisson_ratio, int gauss_order = 7,
    double tanh_precision = 0.01, double tanh_bound = 3.0);

} // namespace tetra
