#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double EDGE_TOLERANCE = 2.2e-10;

struct Vec3 {
    double x{};
    double y{};
    double z{};
};

Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }

Vec3 operator*(double scale, const Vec3& value) {
    return {scale * value.x, scale * value.y, scale * value.z};
}

Vec3 operator/(const Vec3& value, double scale) {
    return {value.x / scale, value.y / scale, value.z / scale};
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm(const Vec3& value) { return std::sqrt(dot(value, value)); }

Vec3 normalized(const Vec3& value, const char* description) {
    const double length = norm(value);
    if (!(length > 0.0)) {
        throw std::invalid_argument(std::string(description) + " must be nonzero");
    }
    return value / length;
}

double clamp_acos_argument(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

struct TriangleFrame {
    Vec3 normal;
    Vec3 strike;
    Vec3 dip;
    Vec3 p1;
    Vec3 p2;
    Vec3 p3;
    Vec3 e12;
    Vec3 e13;
    Vec3 e23;
    double angle_a{};
    double angle_b{};
    double angle_c{};
};

TriangleFrame make_triangle_frame(const Vec3& P1, const Vec3& P2, const Vec3& P3) {
    TriangleFrame frame;
    frame.normal = normalized(cross(P2 - P1, P3 - P1), "triangle normal");
    frame.strike = cross({0.0, 0.0, 1.0}, frame.normal);
    if (norm(frame.strike) == 0.0) {
        frame.strike = {frame.normal.z, 0.0, 0.0};
    }
    frame.strike = normalized(frame.strike, "triangle strike vector");
    frame.dip = cross(frame.normal, frame.strike);

    const auto to_tdcs = [&](const Vec3& value) {
        return Vec3{dot(value, frame.normal), dot(value, frame.strike), dot(value, frame.dip)};
    };
    frame.p1 = to_tdcs(P1 - P2);
    frame.p2 = {0.0, 0.0, 0.0};
    frame.p3 = to_tdcs(P3 - P2);
    frame.e12 = normalized(frame.p2 - frame.p1, "triangle side P1-P2");
    frame.e13 = normalized(frame.p3 - frame.p1, "triangle side P1-P3");
    frame.e23 = normalized(frame.p3 - frame.p2, "triangle side P2-P3");
    frame.angle_a = std::acos(clamp_acos_argument(dot(frame.e12, frame.e13)));
    frame.angle_b = std::acos(clamp_acos_argument(dot(-frame.e12, frame.e23)));
    frame.angle_c = std::acos(clamp_acos_argument(dot(frame.e23, frame.e13)));
    return frame;
}

Vec3 to_tdcs(const Vec3& value, const TriangleFrame& frame) {
    return {dot(value, frame.normal), dot(value, frame.strike), dot(value, frame.dip)};
}

Vec3 from_tdcs(const Vec3& value, const TriangleFrame& frame) {
    return value.x * frame.normal + value.y * frame.strike + value.z * frame.dip;
}

int trimodefinder(const Vec3& coord, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    const double xp = coord.y;
    const double yp = coord.z;
    const double zp = coord.x;
    const double x1 = p1.y, x2 = p2.y, x3 = p3.y;
    const double y1 = p1.z, y2 = p2.z, y3 = p3.z;
    const double denominator = (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
    if (denominator == 0.0) {
        throw std::invalid_argument("degenerate triangle in TDCS");
    }

    const double a = ((xp - x3) * (y2 - y3) - (x2 - x3) * (yp - y3)) / denominator;
    const double b = ((x1 - x3) * (yp - y3) - (xp - x3) * (y1 - y3)) / denominator;
    const double c = 1.0 - a - b;

    int mode = 1;
    if ((a <= 0.0 && b > c && c > a) ||
        (b <= 0.0 && c > a && a > b) ||
        (c <= 0.0 && a > b && b > c)) {
        mode = -1;
    }
    if ((a == 0.0 && b >= 0.0 && c >= 0.0) ||
        (a >= 0.0 && b == 0.0 && c >= 0.0) ||
        (a >= 0.0 && b >= 0.0 && c == 0.0)) {
        mode = 0;
    }
    if (mode == 0 && zp != 0.0) {
        mode = 1;
    }
    return mode;
}

Vec3 ang_dis_disp(const Vec3& coord, double alpha, double bx, double by, double bz, double nu) {
    const double cosA = std::cos(alpha);
    const double sinA = std::sin(alpha);
    const double x = coord.x;
    const double y = coord.y;
    double z = coord.z;
    const double eta = y * cosA - z * sinA;
    double zeta = y * sinA + z * cosA;
    const double r = norm(coord);

    zeta = std::min(zeta, r);
    z = std::min(z, r);
    const double factor = 1.0 / (8.0 * PI * (1.0 - nu));

    const double ux = bx * factor * (x * y / r / (r - z) - x * eta / r / (r - zeta));
    const double vx = bx * factor *
        (eta * sinA / (r - zeta) - y * eta / r / (r - zeta) + y * y / r / (r - z) +
         (1.0 - 2.0 * nu) * (cosA * std::log(r - zeta) - std::log(r - z)));
    const double wx = bx * factor *
        (eta * cosA / (r - zeta) - y / r - eta * z / r / (r - zeta) -
         (1.0 - 2.0 * nu) * sinA * std::log(r - zeta));
    const double uy = by * factor *
        (x * x * cosA / r / (r - zeta) - x * x / r / (r - z) -
         (1.0 - 2.0 * nu) * (cosA * std::log(r - zeta) - std::log(r - z)));
    const double vy = by * x * factor *
        (y * cosA / r / (r - zeta) - sinA * cosA / (r - zeta) - y / r / (r - z));
    const double wy = by * x * factor *
        (z * cosA / r / (r - zeta) - cosA * cosA / (r - zeta) + 1.0 / r);
    const double uz = bz * sinA * factor *
        ((1.0 - 2.0 * nu) * std::log(r - zeta) - x * x / r / (r - zeta));
    const double vz = bz * x * sinA * factor * (sinA / (r - zeta) - y / r / (r - zeta));
    const double wz = bz * x * sinA * factor * (cosA / (r - zeta) - z / r / (r - zeta));
    return {ux + uy + uz, vx + vy + vz, wx + wy + wz};
}

Vec3 td_setup_d(const Vec3& coord, double alpha, double bx, double by, double bz,
                double nu, const Vec3& vertex, const Vec3& side) {
    const double a00 = side.z;
    const double a01 = -side.y;
    const double a10 = side.y;
    const double a11 = side.z;

    Vec3 local = coord;
    const double dy = coord.y - vertex.y;
    const double dz = coord.z - vertex.z;
    local.y = a00 * dy + a01 * dz;
    local.z = a10 * dy + a11 * dz;

    const double local_by = a00 * by + a01 * bz;
    const double local_bz = a10 * by + a11 * bz;
    const Vec3 displacement = ang_dis_disp(local, -PI + alpha, bx, local_by, local_bz, nu);
    return {
        displacement.x,
        displacement.y * a00 + displacement.z * a10,
        displacement.y * a01 + displacement.z * a11,
    };
}

Vec3 td_disp_fs_point(const Vec3& observation, const Vec3& P2, const TriangleFrame& frame,
                      double strike_slip, double dip_slip, double tensile_slip, double nu) {
    const double bx = tensile_slip;
    const double by = strike_slip;
    const double bz = dip_slip;
    const Vec3 x = to_tdcs(observation - P2, frame);
    const int mode = trimodefinder(x, frame.p1, frame.p2, frame.p3);

    Vec3 displacement{};
    if (mode == 1) {
        displacement =
            td_setup_d(x, frame.angle_a, bx, by, bz, nu, frame.p1, -frame.e13) +
            td_setup_d(x, frame.angle_b, bx, by, bz, nu, frame.p2, frame.e12) +
            td_setup_d(x, frame.angle_c, bx, by, bz, nu, frame.p3, frame.e23);
    } else if (mode == -1) {
        displacement =
            td_setup_d(x, frame.angle_a, bx, by, bz, nu, frame.p1, frame.e13) +
            td_setup_d(x, frame.angle_b, bx, by, bz, nu, frame.p2, -frame.e12) +
            td_setup_d(x, frame.angle_c, bx, by, bz, nu, frame.p3, -frame.e23);
    } else {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        displacement = {nan, nan, nan};
    }

    const Vec3 a = frame.p1 - x;
    const Vec3 b = frame.p2 - x;
    const Vec3 c = frame.p3 - x;
    const double na = norm(a), nb = norm(b), nc = norm(c);
    const double numerator = dot(a, cross(b, c));
    const double denominator = na * nb * nc + dot(a, b) * nc + dot(a, c) * nb + dot(b, c) * na;
    const double Fi = -2.0 * std::atan2(numerator, denominator) / (4.0 * PI);
    displacement = displacement + Vec3{bx * Fi, by * Fi, bz * Fi};
    return from_tdcs(displacement, frame);
}

Vec3 ang_dis_disp_fsc(const Vec3& y, double beta, const Vec3& b, double nu, double a) {
    const double sinB = std::sin(beta);
    const double cosB = std::cos(beta);
    const double cotB = 1.0 / std::tan(beta);
    const double y1 = y.x, y2 = y.y, y3 = y.z;
    const double b1 = b.x, b2 = b.y, b3 = b.z;
    const double y3b = y3 + 2.0 * a;
    const double z1b = y1 * cosB + y3b * sinB;
    const double z3b = -y1 * sinB + y3b * cosB;
    const double r2b = y1 * y1 + y2 * y2 + y3b * y3b;
    const double rb = std::sqrt(r2b);
    const double r3b = rb * rb * rb;
    const double Fib = 2.0 * std::atan(-y2 / (-(rb + y3b) / std::tan(beta / 2.0) + y1));
    const double factor = 1.0 / (4.0 * PI * (1.0 - nu));

    const double v1cb1 = b1 * factor *
        (-2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * Fib * cotB * cotB +
         (1.0 - 2.0 * nu) * y2 / (rb + y3b) * ((1.0 - 2.0 * nu - a / rb) * cotB - y1 / (rb + y3b) * (nu + a / rb)) +
         (1.0 - 2.0 * nu) * y2 * cosB * cotB / (rb + z3b) * (cosB + a / rb) +
         a * y2 * (y3b - a) * cotB / r3b +
         y2 * (y3b - a) / (rb * (rb + y3b)) * (-(1.0 - 2.0 * nu) * cotB + y1 / (rb + y3b) * (2.0 * nu + a / rb) + a * y1 / r2b) +
         y2 * (y3b - a) / (rb * (rb + z3b)) *
             (cosB / (rb + z3b) * ((rb * cosB + y3b) * ((1.0 - 2.0 * nu) * cosB - a / rb) * cotB + 2.0 * (1.0 - nu) * (rb * sinB - y1) * cosB) -
              a * y3b * cosB * cotB / r2b));

    const double v2cb1 = b1 * factor *
        ((1.0 - 2.0 * nu) * ((2.0 * (1.0 - nu) * cotB * cotB - nu) * std::log(rb + y3b) -
                              (2.0 * (1.0 - nu) * cotB * cotB + 1.0 - 2.0 * nu) * cosB * std::log(rb + z3b)) -
         (1.0 - 2.0 * nu) / (rb + y3b) * (y1 * cotB * (1.0 - 2.0 * nu - a / rb) + nu * y3b - a + y2 * y2 / (rb + y3b) * (nu + a / rb)) -
         (1.0 - 2.0 * nu) * z1b * cotB / (rb + z3b) * (cosB + a / rb) -
         a * y1 * (y3b - a) * cotB / r3b +
         (y3b - a) / (rb + y3b) * (-2.0 * nu + 1.0 / rb * ((1.0 - 2.0 * nu) * y1 * cotB - a) + y2 * y2 / (rb * (rb + y3b)) * (2.0 * nu + a / rb) + a * y2 * y2 / r3b) +
         (y3b - a) / (rb + z3b) *
             (cosB * cosB - 1.0 / rb * ((1.0 - 2.0 * nu) * z1b * cotB + a * cosB) + a * y3b * z1b * cotB / r3b -
              1.0 / (rb * (rb + z3b)) * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * (rb * cosB + y3b))));

    const double v3cb1 = b1 * factor *
        (2.0 * (1.0 - nu) * ((1.0 - 2.0 * nu) * Fib * cotB + y2 / (rb + y3b) * (2.0 * nu + a / rb) - y2 * cosB / (rb + z3b) * (cosB + a / rb)) +
         y2 * (y3b - a) / rb * (2.0 * nu / (rb + y3b) + a / (rb * rb)) +
         y2 * (y3b - a) * cosB / (rb * (rb + z3b)) *
             (1.0 - 2.0 * nu - (rb * cosB + y3b) / (rb + z3b) * (cosB + a / rb) - a * y3b / (rb * rb)));

    const double v1cb2 = b2 * factor *
        ((1.0 - 2.0 * nu) * ((2.0 * (1.0 - nu) * cotB * cotB + nu) * std::log(rb + y3b) -
                              (2.0 * (1.0 - nu) * cotB * cotB + 1.0) * cosB * std::log(rb + z3b)) +
         (1.0 - 2.0 * nu) / (rb + y3b) * (-(1.0 - 2.0 * nu) * y1 * cotB + nu * y3b - a + a * y1 * cotB / rb + y1 * y1 / (rb + y3b) * (nu + a / rb)) -
         (1.0 - 2.0 * nu) * cotB / (rb + z3b) * (z1b * cosB - a * (rb * sinB - y1) / (rb * cosB)) -
         a * y1 * (y3b - a) * cotB / r3b +
         (y3b - a) / (rb + y3b) * (2.0 * nu + 1.0 / rb * ((1.0 - 2.0 * nu) * y1 * cotB + a) - y1 * y1 / (rb * (rb + y3b)) * (2.0 * nu + a / rb) - a * y1 * y1 / r3b) +
         (y3b - a) * cotB / (rb + z3b) *
             (-cosB * sinB + a * y1 * y3b / (r3b * cosB) + (rb * sinB - y1) / rb *
                 (2.0 * (1.0 - nu) * cosB - (rb * cosB + y3b) / (rb + z3b) * (1.0 + a / (rb * cosB)))));

    const double v2cb2 = b2 * factor *
        (2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * Fib * cotB * cotB +
         (1.0 - 2.0 * nu) * y2 / (rb + y3b) * (-(1.0 - 2.0 * nu - a / rb) * cotB + y1 / (rb + y3b) * (nu + a / rb)) -
         (1.0 - 2.0 * nu) * y2 * cotB / (rb + z3b) * (1.0 + a / (rb * cosB)) -
         a * y2 * (y3b - a) * cotB / r3b +
         y2 * (y3b - a) / (rb * (rb + y3b)) * ((1.0 - 2.0 * nu) * cotB - 2.0 * nu * y1 / (rb + y3b) - a * y1 / rb * (1.0 / rb + 1.0 / (rb + y3b))) +
         y2 * (y3b - a) * cotB / (rb * (rb + z3b)) *
             (-2.0 * (1.0 - nu) * cosB + (rb * cosB + y3b) / (rb + z3b) * (1.0 + a / (rb * cosB)) + a * y3b / (r2b * cosB)));

    const double v3cb2 = b2 * factor *
        (-2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * cotB * (std::log(rb + y3b) - cosB * std::log(rb + z3b)) -
         2.0 * (1.0 - nu) * y1 / (rb + y3b) * (2.0 * nu + a / rb) +
         2.0 * (1.0 - nu) * z1b / (rb + z3b) * (cosB + a / rb) +
         (y3b - a) / rb * ((1.0 - 2.0 * nu) * cotB - 2.0 * nu * y1 / (rb + y3b) - a * y1 / r2b) -
         (y3b - a) / (rb + z3b) *
             (cosB * sinB + (rb * cosB + y3b) * cotB / rb * (2.0 * (1.0 - nu) * cosB - (rb * cosB + y3b) / (rb + z3b)) +
              a / rb * (sinB - y3b * z1b / r2b - z1b * (rb * cosB + y3b) / (rb * (rb + z3b)))));

    const double v1cb3 = b3 * factor *
        ((1.0 - 2.0 * nu) * (y2 / (rb + y3b) * (1.0 + a / rb) - y2 * cosB / (rb + z3b) * (cosB + a / rb)) -
         y2 * (y3b - a) / rb * (a / r2b + 1.0 / (rb + y3b)) +
         y2 * (y3b - a) * cosB / (rb * (rb + z3b)) * ((rb * cosB + y3b) / (rb + z3b) * (cosB + a / rb) + a * y3b / r2b));

    const double v2cb3 = b3 * factor *
        ((1.0 - 2.0 * nu) * (-sinB * std::log(rb + z3b) - y1 / (rb + y3b) * (1.0 + a / rb) + z1b / (rb + z3b) * (cosB + a / rb)) +
         y1 * (y3b - a) / rb * (a / r2b + 1.0 / (rb + y3b)) -
         (y3b - a) / (rb + z3b) *
             (sinB * (cosB - a / rb) + z1b / rb * (1.0 + a * y3b / r2b) -
              1.0 / (rb * (rb + z3b)) * (y2 * y2 * cosB * sinB - a * z1b / rb * (rb * cosB + y3b))));

    const double v3cb3 = b3 * factor *
        (2.0 * (1.0 - nu) * Fib + 2.0 * (1.0 - nu) * y2 * sinB / (rb + z3b) * (cosB + a / rb) +
         y2 * (y3b - a) * sinB / (rb * (rb + z3b)) * (1.0 + (rb * cosB + y3b) / (rb + z3b) * (cosB + a / rb) + a * y3b / r2b));

    return {v1cb1 + v1cb2 + v1cb3, v2cb1 + v2cb2 + v2cb3, v3cb1 + v3cb2 + v3cb3};
}

// Exact row-vector matrix operations used by the original NumPy implementation.
Vec3 row_times_rows_matrix(const Vec3& value, const std::array<Vec3, 3>& rows) {
    return {
        value.x * rows[0].x + value.y * rows[1].x + value.z * rows[2].x,
        value.x * rows[0].y + value.y * rows[1].y + value.z * rows[2].y,
        value.x * rows[0].z + value.y * rows[1].z + value.z * rows[2].z,
    };
}

Vec3 row_times_transpose(const Vec3& value, const std::array<Vec3, 3>& rows) {
    return {dot(value, rows[0]), dot(value, rows[1]), dot(value, rows[2])};
}

Vec3 ang_setup_fsc_point(const Vec3& observation, const Vec3& burgers,
                         const Vec3& PA, const Vec3& PB, double nu) {
    const Vec3 side = PB - PA;
    const double side_length = norm(side);
    if (!(side_length > 0.0)) {
        throw std::invalid_argument("triangle edge must be nonzero");
    }
    const double beta = std::acos(clamp_acos_argument(-side.z / side_length));
    if (std::abs(beta) < EDGE_TOLERANCE || std::abs(PI - beta) < EDGE_TOLERANCE) {
        return {};
    }

    Vec3 ey1{side.x, side.y, 0.0};
    ey1 = normalized(ey1, "horizontal edge projection");
    const Vec3 ey3{0.0, 0.0, -1.0};
    const Vec3 ey2 = cross(ey3, ey1);
    const std::array<Vec3, 3> axes{ey1, ey2, ey3};

    const Vec3 yA = row_times_rows_matrix(observation - PA, axes);
    const Vec3 yAB = row_times_rows_matrix(side, axes);
    const Vec3 yB = yA - yAB;
    const Vec3 local_burgers = row_times_rows_matrix(burgers, axes);

    const double configuration_beta = beta * yA.x >= 0.0 ? -PI + beta : beta;
    const Vec3 vA = ang_dis_disp_fsc(yA, configuration_beta, local_burgers, nu, -PA.z);
    const Vec3 vB = ang_dis_disp_fsc(yB, configuration_beta, local_burgers, nu, -PB.z);
    return row_times_transpose(vB - vA, axes);
}

Vec3 td_disp_harmonic_point(const Vec3& observation, const Vec3& P1, const Vec3& P2,
                            const Vec3& P3, const TriangleFrame& frame,
                            double strike_slip, double dip_slip, double tensile_slip, double nu) {
    const Vec3 burgers = tensile_slip * frame.normal + strike_slip * frame.strike + dip_slip * frame.dip;
    return ang_setup_fsc_point(observation, burgers, P1, P2, nu) +
           ang_setup_fsc_point(observation, burgers, P2, P3, nu) +
           ang_setup_fsc_point(observation, burgers, P3, P1, nu);
}

enum class SolutionKind { FullSpace, Harmonic, HalfSpace };

std::vector<Vec3> calculate_displacement(
    const std::vector<Vec3>& observations, Vec3 P1, Vec3 P2, Vec3 P3,
    double strike_slip, double dip_slip, double tensile_slip, double nu,
    SolutionKind kind) {
    if (!(nu > -1.0 && nu < 0.5)) {
        throw std::invalid_argument("Poisson ratio must be in (-1, 0.5)");
    }
    if (kind == SolutionKind::HalfSpace) {
        for (const Vec3& point : observations) {
            if (point.z > 0.0) {
                throw std::invalid_argument("half-space observation Z coordinates must be nonpositive");
            }
        }
        if (P1.z > 0.0 || P2.z > 0.0 || P3.z > 0.0) {
            throw std::invalid_argument("half-space triangle Z coordinates must be nonpositive");
        }
    }

    const TriangleFrame frame = make_triangle_frame(P1, P2, P3);
    std::vector<Vec3> result(observations.size());

    if (kind == SolutionKind::FullSpace) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(observations.size()); ++i) {
            result[static_cast<std::size_t>(i)] = td_disp_fs_point(
                observations[static_cast<std::size_t>(i)], P2, frame,
                strike_slip, dip_slip, tensile_slip, nu);
        }
        return result;
    }

    if (kind == SolutionKind::Harmonic) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(observations.size()); ++i) {
            result[static_cast<std::size_t>(i)] = td_disp_harmonic_point(
                observations[static_cast<std::size_t>(i)], P1, P2, P3, frame,
                strike_slip, dip_slip, tensile_slip, nu);
        }
        return result;
    }

    const Vec3 image_P1{P1.x, P1.y, -P1.z};
    const Vec3 image_P2{P2.x, P2.y, -P2.z};
    const Vec3 image_P3{P3.x, P3.y, -P3.z};
    const TriangleFrame image_frame = make_triangle_frame(image_P1, image_P2, image_P3);
    const bool surface_triangle = image_P1.z == 0.0 && image_P2.z == 0.0 && image_P3.z == 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(observations.size()); ++i) {
        const Vec3& observation = observations[static_cast<std::size_t>(i)];
        const Vec3 main = td_disp_fs_point(observation, P2, frame, strike_slip, dip_slip, tensile_slip, nu);
        const Vec3 harmonic = td_disp_harmonic_point(
            observation, P1, P2, P3, frame, strike_slip, dip_slip, tensile_slip, nu);
        Vec3 image = td_disp_fs_point(
            observation, image_P2, image_frame, strike_slip, dip_slip, tensile_slip, nu);
        if (surface_triangle) {
            image.z = -image.z;
        }
        Vec3 total = main + image + harmonic;
        if (surface_triangle) {
            total = -total;
        }
        result[static_cast<std::size_t>(i)] = total;
    }
    return result;
}

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

Vec3 read_vertex(const Array& vertex, const char* name) {
    const auto info = vertex.request();
    if (info.ndim != 1 || info.shape[0] != 3) {
        throw py::value_error(std::string(name) + " must have shape (3,)");
    }
    const auto* values = static_cast<const double*>(info.ptr);
    return {values[0], values[1], values[2]};
}

py::tuple displacement_py(
    const Array& X, const Array& Y, const Array& Z,
    const Array& P1_array, const Array& P2_array, const Array& P3_array,
    double strike_slip, double dip_slip, double tensile_slip, double nu,
    SolutionKind kind) {
    const auto x = X.request();
    const auto y = Y.request();
    const auto z = Z.request();
    if (x.shape != y.shape || x.shape != z.shape) {
        throw py::value_error("X, Y, and Z must have identical shapes");
    }

    const auto* x_values = static_cast<const double*>(x.ptr);
    const auto* y_values = static_cast<const double*>(y.ptr);
    const auto* z_values = static_cast<const double*>(z.ptr);
    std::vector<Vec3> observations(static_cast<std::size_t>(x.size));
    for (py::ssize_t i = 0; i < x.size; ++i) {
        observations[static_cast<std::size_t>(i)] = {x_values[i], y_values[i], z_values[i]};
    }

    const Vec3 P1 = read_vertex(P1_array, "P1");
    const Vec3 P2 = read_vertex(P2_array, "P2");
    const Vec3 P3 = read_vertex(P3_array, "P3");
    std::vector<Vec3> displacement;
    {
        py::gil_scoped_release release;
        displacement = calculate_displacement(
            observations, P1, P2, P3,
            strike_slip, dip_slip, tensile_slip, nu, kind);
    }

    const py::array::ShapeContainer shape(x.shape);
    py::array_t<double> ue(shape), un(shape), uv(shape);
    auto* ue_values = ue.mutable_data();
    auto* un_values = un.mutable_data();
    auto* uv_values = uv.mutable_data();
    for (std::size_t i = 0; i < displacement.size(); ++i) {
        ue_values[i] = displacement[i].x;
        un_values[i] = displacement[i].y;
        uv_values[i] = displacement[i].z;
    }
    return py::make_tuple(std::move(ue), std::move(un), std::move(uv));
}

}  // namespace

PYBIND11_MODULE(dislocation_displacement, module) {
    module.doc() = "Triangular-dislocation displacement solutions in full and elastic half spaces";

    const auto bind = [&](const char* name, SolutionKind kind, const char* description) {
        module.def(
            name,
            [kind](const Array& X, const Array& Y, const Array& Z,
                   const Array& P1, const Array& P2, const Array& P3,
                   double Ss, double Ds, double Ts, double nu) {
                return displacement_py(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu, kind);
            },
            py::arg("X"), py::arg("Y"), py::arg("Z"),
            py::arg("P1"), py::arg("P2"), py::arg("P3"),
            py::arg("Ss"), py::arg("Ds"), py::arg("Ts"), py::arg("nu"),
            description);
    };

    bind("TDdispFS", SolutionKind::FullSpace, "Full-space triangular-dislocation displacement.");
    bind("TDdisp_HarFunc", SolutionKind::Harmonic, "Half-space harmonic correction displacement.");
    bind("TDdispHS", SolutionKind::HalfSpace, "Complete elastic half-space displacement.");
}
