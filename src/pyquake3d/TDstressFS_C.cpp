// ang_dis_strain.cpp
#include <vector>
#include <array>
#include <cmath>

#include <cstddef>
#include <vector>
#include <cmath>
#include <cstring> // for memset
#include <iostream>
#include <algorithm>
using namespace std;

const double PI = 3.141592653589793;

// 3D 向量类型
typedef array<double, 3> Vec3;
using namespace std;


void TensTrans(std::vector<double>& Txx, std::vector<double>& Tyy, std::vector<double>& Tzz,
               std::vector<double>& Txy, std::vector<double>& Txz, std::vector<double>& Tyz,
               const double A[3][3], size_t n_pts) {
    // 
    for (size_t i = 0; i < n_pts; ++i) {
        double Txx2 = 0, Tyy2 = 0, Tzz2 = 0, Txy2 = 0, Txz2 = 0, Tyz2 = 0;
        
        // Txx2, Tyy2, Tzz2, Txy2, Txz2, Tyz2 
        Txx2 = A[0][0] * A[0][0] * Txx[i] + 2 * A[0][0] * A[1][0] * Txy[i] + 2 * A[0][0] * A[2][0] * Txz[i] + 2 * A[1][0] * A[2][0] * Tyz[i] + A[1][0] * A[1][0] * Tyy[i] + A[2][0] * A[2][0] * Tzz[i];
        
        Tyy2 = A[0][1] * A[0][1] * Txx[i] + 2 * A[0][1] * A[1][1] * Txy[i] + 2 * A[0][1] * A[2][1] * Txz[i] + 2 * A[1][1] * A[2][1] * Tyz[i] + A[1][1] * A[1][1] * Tyy[i] + A[2][1] * A[2][1] * Tzz[i];
        
        Tzz2 = A[0][2] * A[0][2] * Txx[i] + 2 * A[0][2] * A[1][2] * Txy[i] + 2 * A[0][2] * A[2][2] * Txz[i] + 2 * A[1][2] * A[2][2] * Tyz[i] + A[1][2] * A[1][2] * Tyy[i] + A[2][2] * A[2][2] * Tzz[i];
        
        Txy2 = A[0][0] * A[0][1] * Txx[i] + (A[0][0] * A[1][1] + A[0][1] * A[1][0]) * Txy[i] + (A[0][0] * A[2][1] + A[0][1] * A[2][0]) * Txz[i] + (A[2][1] * A[1][0] + A[2][0] * A[1][1]) * Tyz[i] + A[1][1] * A[1][0] * Tyy[i] + A[2][0] * A[2][1] * Tzz[i];
        
        Txz2 = A[0][0] * A[0][2] * Txx[i] + (A[0][0] * A[1][2] + A[0][2] * A[1][0]) * Txy[i] + (A[0][0] * A[2][2] + A[0][2] * A[2][0]) * Txz[i] + (A[2][2] * A[1][0] + A[2][0] * A[1][2]) * Tyz[i] + A[1][2] * A[1][0] * Tyy[i] + A[2][0] * A[2][2] * Tzz[i];
        
        Tyz2 = A[0][1] * A[0][2] * Txx[i] + (A[0][2] * A[1][1] + A[0][1] * A[1][2]) * Txy[i] + (A[0][2] * A[2][1] + A[0][1] * A[2][2]) * Txz[i] + (A[2][1] * A[1][2] + A[2][2] * A[1][1]) * Tyz[i] + A[1][1] * A[1][2] * Tyy[i] + A[2][1] * A[2][2] * Tzz[i];
        
        // 
        Txx[i] = Txx2;
        Tyy[i] = Tyy2;
        Tzz[i] = Tzz2;
        Txy[i] = Txy2;
        Txz[i] = Txz2;
        Tyz[i] = Tyz2;
    }
}

void AngDisStrain(
    const std::vector<std::array<double, 3>>& coord,
    double alpha, double bx, double by, double bz, double nu,
    std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
    std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz)
{
    size_t n = coord.size();
    Exx.resize(n); Eyy.resize(n); Ezz.resize(n);
    Exy.resize(n); Exz.resize(n); Eyz.resize(n);

    double cosA = std::cos(alpha);
    double sinA = std::sin(alpha);

    for (size_t i = 0; i < n; ++i) {
        double x = coord[i][0];
        double y = coord[i][1];
        double z = coord[i][2];

        double eta = y * cosA - z * sinA;
        double zeta = y * sinA + z * cosA;

        double x2 = x * x, y2 = y * y, z2 = z * z;
        double r2 = x2 + y2 + z2;
        double r = std::sqrt(r2);
        double r3 = r2 * r;
        double rz = r * (r - z);
        double r2z2 = r2 * (r - z) * (r - z);
        double r3z = r3 * (r - z);

        double W = zeta - r;
        double W2 = W * W;
        double Wr = W * r;
        double W2r = W2 * r;
        double Wr3 = W * r3;
        double W2r2 = W2 * r2;

        double C = (r * cosA - z) / Wr;
        double S = (r * sinA - y) / Wr;

        double rFi_rx = (eta / r / (r - zeta) - y / r / (r - z)) / 4.0 / PI;
        double rFi_ry = (x / r / (r - z) - cosA * x / r / (r - zeta)) / 4.0 / PI;
        double rFi_rz = (sinA * x / r / (r - zeta)) / 4.0 / PI;

        Exx[i] = bx * rFi_rx + bx / 8.0 / PI / (1 - nu) * (eta / Wr + eta * x2 / W2r2 - eta * x2 / Wr3 + y / rz - x2 * y / r2z2 - x2 * y / r3z)
                - by * x / 8.0 / PI / (1 - nu) * (((2 * nu + 1) / Wr + x2 / W2r2 - x2 / Wr3) * cosA + (2 * nu + 1) / rz - x2 / r2z2 - x2 / r3z)
                + bz * x * sinA / 8.0 / PI / (1 - nu) * ((2 * nu + 1) / Wr + x2 / W2r2 - x2 / Wr3);

        Eyy[i] = by * rFi_ry + bx / 8.0 / PI / (1 - nu) * ((1.0 / Wr + S * S - y2 / Wr3) * eta + (2 * nu + 1) * y / rz - y2 * y / r2z2 - y2 * y / r3z - 2 * nu * cosA * S)
                - by * x / 8.0 / PI / (1 - nu) * (1.0 / rz - y2 / r2z2 - y2 / r3z + (1.0 / Wr + S * S - y2 / Wr3) * cosA)
                + bz * x * sinA / 8.0 / PI / (1 - nu) * (1.0 / Wr + S * S - y2 / Wr3);

        Ezz[i] = bz * rFi_rz + bx / 8.0 / PI / (1 - nu) * (eta / W / r + eta * C * C - eta * z2 / Wr3 + y * z / r3 + 2 * nu * sinA * C)
                - by * x / 8.0 / PI / (1 - nu) * ((1.0 / Wr + C * C - z2 / Wr3) * cosA + z / r3)
                + bz * x * sinA / 8.0 / PI / (1 - nu) * (1.0 / Wr + C * C - z2 / Wr3);

        Exy[i] = bx * rFi_ry / 2 + by * rFi_rx / 2
                - bx / 8.0 / PI / (1 - nu) * (x * y2 / r2z2 - nu * x / rz + x * y2 / r3z - nu * x * cosA / Wr + eta * x * S / Wr + eta * x * y / Wr3)
                + by / 8.0 / PI / (1 - nu) * (x2 * y / r2z2 - nu * y / rz + x2 * y / r3z + nu * cosA * S + x2 * y * cosA / Wr3 + x2 * cosA * S / Wr)
                - bz * sinA / 8.0 / PI / (1 - nu) * (nu * S + x2 * S / Wr + x2 * y / Wr3);

        Exz[i] = bx * rFi_rz / 2 + bz * rFi_rx / 2
                - bx / 8.0 / PI / (1 - nu) * (-x * y / r3 + nu * x * sinA / Wr + eta * x * C / Wr + eta * x * z / Wr3)
                + by / 8.0 / PI / (1 - nu) * (-x2 / r3 + nu / r + nu * cosA * C + x2 * z * cosA / Wr3 + x2 * cosA * C / Wr)
                - bz * sinA / 8.0 / PI / (1 - nu) * (nu * C + x2 * C / Wr + x2 * z / Wr3);

        Eyz[i] = by * rFi_rz / 2 + bz * rFi_ry / 2
                + bx / 8.0 / PI / (1 - nu) * (y2 / r3 - nu / r - nu * cosA * C + nu * sinA * S + eta * sinA * cosA / W2 - eta * (y * cosA + z * sinA) / W2r + eta * y * z / W2r2 - eta * y * z / Wr3)
                - by * x / 8.0 / PI / (1 - nu) * (y / r3 + sinA * cosA * cosA / W2 - cosA * (y * cosA + z * sinA) / W2r + y * z * cosA / W2r2 - y * z * cosA / Wr3)
                - bz * x * sinA / 8.0 / PI / (1 - nu) * (y * z / Wr3 - sinA * cosA / W2 + (y * cosA + z * sinA) / W2r - y * z / W2r2);
    }
}



std::vector<int> trimodefinder(
    const std::vector<std::array<double, 3>>& coord, // 
    const std::vector<double>& p1,
    const std::vector<double>& p2,
    const std::vector<double>& p3
) {
    size_t N = coord.size();
    std::vector<int> trimode(N, 1);  // 

    double x1 = p1[1], x2 = p2[1], x3 = p3[1];
    double y1 = p1[2], y2 = p2[2], y3 = p3[2];

    double denom = (x1 - x3)*(y2 - y3) - (x2 - x3)*(y1 - y3);

    for (size_t i = 0; i < N; ++i) {
        double xp = coord[i][1];
        double yp = coord[i][2];
        double zp = coord[i][0];

        double a = ((xp - x3)*(y2 - y3) - (x2 - x3)*(yp - y3)) / denom;
        double b = ((x1 - x3)*(yp - y3) - (xp - x3)*(y1 - y3)) / denom;
        double c = 1.0 - a - b;

        if ((a <= 0) && (b > c) && (c > a))
            trimode[i] = -1;
        else if ((b <= 0) && (c > a) && (a > b))
            trimode[i] = -1;
        else if ((c <= 0) && (a > b) && (b > c))
            trimode[i] = -1;
        else if ((a == 0) && (b >= 0) && (c >= 0))
            trimode[i] = 0;
        else if ((a >= 0) && (b == 0) && (c >= 0))
            trimode[i] = 0;
        else if ((a >= 0) && (b >= 0) && (c == 0))
            trimode[i] = 0;

        if (trimode[i] == 0 && zp != 0)
            trimode[i] = 1;
    }

    return trimode;
}


void mat2x2_mul_vec2(const double A[2][2], const double v[2], double result[2]) {
    result[0] = A[0][0] * v[0] + A[0][1] * v[1];
    result[1] = A[1][0] * v[0] + A[1][1] * v[1];
}

void vec2_mul_mat2x2(const double A[2][2], const double v[2], double result[2]) {
    result[0] = A[0][0] * v[0] + A[1][0] * v[1];
    result[1] = A[0][1] * v[0] + A[1][1] * v[1];
}

void mat2x2_transpose(const double A[2][2], double AT[2][2]) {
    AT[0][0] = A[0][0]; AT[0][1] = A[1][0];
    AT[1][0] = A[0][1]; AT[1][1] = A[1][1];
}

// 
void TDSetupS(const std::vector<std::array<double, 3>>& coord,
              double alpha, double bx, double by, double bz,
              double nu,
              const std::vector<double> TriVertex,
              const std::vector<double> SideVec,
              std::vector<double>& Exx,
              std::vector<double>& Eyy,
              std::vector<double>& Ezz,
              std::vector<double>& Exy,
              std::vector<double>& Exz,
              std::vector<double>& Eyz) {
    
    size_t n_pts = coord.size();

    // 
    double A[2][2] = {
        { SideVec[2], -SideVec[1] },
        { SideVec[1],  SideVec[2] }
    };
	
    // r1 = [(y - y0, z - z0) * A^T]
    std::vector<std::array<double, 3>> coord1 = coord;
    double AT[2][2];
    mat2x2_transpose(A, AT);

    for (size_t i = 0; i < n_pts; ++i) {
        double r_in[2] = { coord[i][1] - TriVertex[1], coord[i][2] - TriVertex[2] };
		//printf("%f %f %f %f\n",r_in[0],r_in[1],coord[i][1],TriVertex[1]);
        double r_out[2];
        vec2_mul_mat2x2(AT, r_in, r_out);
        coord1[i][1] = r_out[0]; // new y
        coord1[i][2] = r_out[1]; // new z
    }
	
    // 
    double slip_in[2] = { by, bz };
    double slip_out[2];
    mat2x2_mul_vec2(A, slip_in, slip_out);
    double by1 = slip_out[0];
    double bz1 = slip_out[1];

    // 
    Exx.resize(n_pts);
    Eyy.resize(n_pts);
    Ezz.resize(n_pts);
    Exy.resize(n_pts);
    Exz.resize(n_pts);
    Eyz.resize(n_pts);
	//printf("%f %f %f %f %f %f %f\n",coord1[0][0],coord1[0][1],coord1[0][2],-M_PI + alpha,bx, by1, bz1);
    // 
    AngDisStrain(coord1, -PI + alpha, bx, by1, bz1, nu,
                 Exx, Eyy, Ezz, Exy, Exz, Eyz);
	//printf("%.10f %.10f %.10f %.10f %.10f %.10f\n",Exx[0],Eyy[0],Ezz[0],Exy[0],Exz[0],Eyz[0]);
    // 
    double B[3][3] = {0};
    B[0][0] = 1.0;
    B[1][1] = A[0][0];
    B[1][2] = A[0][1];
    B[2][1] = A[1][0];
    B[2][2] = A[1][1];
	/*for(size_t i = 0; i < 3; ++i)
	{
		for(size_t j = 0; j < 3; ++j)
		{
			printf("%f ",B[i][j]);
		}
		printf("\n");
	}*/
    // 
    TensTrans(Exx, Eyy, Ezz, Exy, Exz, Eyz, B, n_pts);
	//printf("%.10f %.10f %.10f %.10f %.10f %.10f\n",Exx[0], Eyy[0], Ezz[0], Exy[0], Exz[0], Eyz[0]);
}










// 
std::vector<double> vectorOpposite(const std::vector<double>& v) {
    return {-v[0], -v[1], -v[2]};
}


// Main function to compute stress and strain
void TDstressFS(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z,
                const std::vector<double>& P1, const std::vector<double>& P2, const std::vector<double>& P3,
                double Ss, double Ds, double Ts, double mu, double lambda_,
                std::vector<double>& Sxx, std::vector<double>& Syy, std::vector<double>& Szz,
                std::vector<double>& Sxy, std::vector<double>& Sxz, std::vector<double>& Syz,
                std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
                std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz) 
{
    size_t n = X.size();
	//printf("n= %d",n);
    std::vector<std::vector<double>> coords(n, std::vector<double>(3));
    for (size_t i = 0; i < n; ++i) {
        coords[i] = {X[i], Y[i], Z[i]};
    }

    double bx = Ts;
    double by = Ss;
    double bz = Ds;
    double nu = 1.0 / (1.0 + lambda_ / mu) / 2.0;

    // 
    std::vector<double> Vnorm(3);
    //Vnorm[0] = P2[1] * P3[2] - P2[2] * P3[1];
    //Vnorm[1] = P2[2] * P3[0] - P2[0] * P3[2];
    //Vnorm[2] = P2[0] * P3[1] - P2[1] * P3[0];
	Vnorm[0]=(P2[1]-P1[1])*(P3[2]-P1[2])-(P2[2]-P1[2])*(P3[1]-P1[1]);
	Vnorm[1]=(P2[2]-P1[2])*(P3[0]-P1[0])-(P2[0]-P1[0])*(P3[2]-P1[2]);
	Vnorm[2]=(P2[0]-P1[0])*(P3[1]-P1[1])-(P2[1]-P1[1])*(P3[0]-P1[0]);
	
    double norm = std::sqrt(Vnorm[0] * Vnorm[0] + Vnorm[1] * Vnorm[1] + Vnorm[2] * Vnorm[2]);
    Vnorm[0] /= norm;
    Vnorm[1] /= norm;
    Vnorm[2] /= norm;
    std::vector<double> eX = {1, 0, 0};
    std::vector<double> eY = {0, 1, 0};
    std::vector<double> eZ = {0, 0, 1};
    
    //  Vstrike
    std::vector<double> Vstrike(3);
    Vstrike[0] = eZ[1] * Vnorm[2] - eZ[2] * Vnorm[1];
    Vstrike[1] = eZ[2] * Vnorm[0] - eZ[0] * Vnorm[2];
    Vstrike[2] = eZ[0] * Vnorm[1] - eZ[1] * Vnorm[0];
	
    norm = std::sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] + Vstrike[2] * Vstrike[2]);
    Vstrike[0] /= norm;
    Vstrike[1] /= norm;
    Vstrike[2] /= norm;
    if (norm <1e-5) {
        Vstrike[0] = eX[0] * Vnorm[2];
        Vstrike[1] = eX[1] * Vnorm[2];
        Vstrike[2] = eX[2] * Vnorm[2];
    }
    
    
	
    //  Vdip
    std::vector<double> Vdip(3);
    Vdip[0] = Vnorm[1] * Vstrike[2] - Vnorm[2] * Vstrike[1];
    Vdip[1] = Vnorm[2] * Vstrike[0] - Vnorm[0] * Vstrike[2];
    Vdip[2] = Vnorm[0] * Vstrike[1] - Vnorm[1] * Vstrike[0];
	
    // Construct transformation matrix At
	double At[3][3] = {0};
    //std::vector<std::vector<double>> At(3, std::vector<double>(3));
    At[0][0] = Vnorm[0]; At[0][1] = Vstrike[0]; At[0][2] = Vdip[0];
    At[1][0] = Vnorm[1]; At[1][1] = Vstrike[1]; At[1][2] = Vdip[1];
    At[2][0] = Vnorm[2]; At[2][1] = Vstrike[2]; At[2][2] = Vdip[2];
	
	
    // Transform coordinates
    //std::vector<std::vector<double>> X1(n, std::vector<double>(3));
	std::vector<std::array<double, 3>> X1(n);  // 

	for (size_t i = 0; i < n; ++i) {
		std::vector<double> delta = {
			coords[i][0] - P2[0],
			coords[i][1] - P2[1],
			coords[i][2] - P2[2]
		};

		for (size_t j = 0; j < 3; ++j) {
			X1[i][j] = At[0][j] * delta[0] + At[1][j] * delta[1] + At[2][j] * delta[2];
		}
		//printf("%f %f %f\n",delta[0],delta[1],delta[2]);
	}
	//printf("%f %f %f\n",P1[0],P1[1],P1[2]);
	//printf("%f %f %f\n",P2[0],P2[1],P2[2]);
	//printf("%f %f %f\n",P3[0],P3[1],P3[2]);
	
    // Transform triangle vertices
    std::vector<double> p1(3), p2(3), p3(3);
    for (size_t i = 0; i < 3; ++i) {
        p1[i] = At[0][i] * (P1[0] - P2[0]) + At[1][i] * (P1[1] - P2[1]) + At[2][i] * (P1[2] - P2[2]);
        p2[i] = 0.0; // Since P2 - P2 = 0
        p3[i] = At[0][i] * (P3[0] - P2[0]) + At[1][i] * (P3[1] - P2[1]) + At[2][i] * (P3[2] - P2[2]);
    }
	
	// for(size_t i = 0; i < 3; ++i)
	// {
	// 	for(size_t j = 0; j < 3; ++j)
	// 	{
	// 		printf("%f ",At[i][j]);
	// 	}
	// 	printf("\n");
	// }
	
    // e12, e13, e23
    std::vector<double> e12 = {(p2[0] - p1[0]), (p2[1] - p1[1]), (p2[2] - p1[2])};
    std::vector<double> e13 = {(p3[0] - p1[0]), (p3[1] - p1[1]), (p3[2] - p1[2])};
    std::vector<double> e23 = {(p3[0] - p2[0]), (p3[1] - p2[1]), (p3[2] - p2[2])};

    double norm_e12 = std::sqrt(e12[0] * e12[0] + e12[1] * e12[1] + e12[2] * e12[2]);
    e12[0] /= norm_e12; e12[1] /= norm_e12; e12[2] /= norm_e12;

    double norm_e13 = std::sqrt(e13[0] * e13[0] + e13[1] * e13[1] + e13[2] * e13[2]);
    e13[0] /= norm_e13; e13[1] /= norm_e13; e13[2] /= norm_e13;

    double norm_e23 = std::sqrt(e23[0] * e23[0] + e23[1] * e23[1] + e23[2] * e23[2]);
    e23[0] /= norm_e23; e23[1] /= norm_e23; e23[2] /= norm_e23;

    // A_angle, B_angle, C_angle
    double A_angle = std::acos(e12[0] * e13[0] + e12[1] * e13[1] + e12[2] * e13[2]);
    double B_angle = std::acos((-e12[0]) * e23[0] + (-e12[1]) * e23[1] + (-e12[2]) * e23[2]);
    double C_angle = std::acos(e23[0] * e13[0] + e23[1] * e13[1] + e23[2] * e13[2]);
	
	std::vector<int> Trimode = trimodefinder(X1, p1, p2, p3);
	
	//printf("%f %f %f\n",A_angle,B_angle,C_angle);
    
    Exx.resize(n, 0.0);
    Eyy.resize(n, 0.0);
    Ezz.resize(n, 0.0);
    Exy.resize(n, 0.0);
    Exz.resize(n, 0.0);
    Eyz.resize(n, 0.0);
	
	std::vector<bool> casepLog(X1.size()), casenLog(X1.size()), casezLog(X1.size());
	for (size_t i = 0; i < X1.size(); ++i) {
		casepLog[i] = (Trimode[i] == 1);
		casenLog[i] = (Trimode[i] == -1);
		casezLog[i] = (Trimode[i] == 0);
	}
	// 
	std::vector<std::array<double, 3>> Xp, Xn;
	for (size_t i = 0; i < X1.size(); ++i) {
		if (casepLog[i]) Xp.push_back(X1[i]);
		else if (casenLog[i]) Xn.push_back(X1[i]);
	}
	//
	std::vector<double> Exx1Tp, Eyy1Tp, Ezz1Tp, Exy1Tp, Exz1Tp, Eyz1Tp;
	std::vector<double> Exx2Tp, Eyy2Tp, Ezz2Tp, Exy2Tp, Exz2Tp, Eyz2Tp;
	std::vector<double> Exx3Tp, Eyy3Tp, Ezz3Tp, Exy3Tp, Exz3Tp, Eyz3Tp;
	std::vector<double> Exx1Tn, Eyy1Tn, Ezz1Tn, Exy1Tn, Exz1Tn, Eyz1Tn;
	std::vector<double> Exx2Tn, Eyy2Tn, Ezz2Tn, Exy2Tn, Exz2Tn, Eyz2Tn;
	std::vector<double> Exx3Tn, Eyy3Tn, Ezz3Tn, Exy3Tn, Exz3Tn, Eyz3Tn;
	
	//printf("%f %f %f %f %f %f %f %f %f %f\n",Xp[0][0],Xp[0][1],Xp[0][2],B_angle, bx, by, bz, nu, p2[2], e12[2]);
	
	if (!Xp.empty()) {
		TDSetupS(Xp, A_angle, bx, by, bz, nu, p1, vectorOpposite(e13), Exx1Tp, Eyy1Tp, Ezz1Tp, Exy1Tp, Exz1Tp, Eyz1Tp);
		TDSetupS(Xp, B_angle, bx, by, bz, nu, p2, e12, Exx2Tp, Eyy2Tp, Ezz2Tp, Exy2Tp, Exz2Tp, Eyz2Tp);
		TDSetupS(Xp, C_angle, bx, by, bz, nu, p3, e23, Exx3Tp, Eyy3Tp, Ezz3Tp, Exy3Tp, Exz3Tp, Eyz3Tp);
	}

	if (!Xn.empty()) {
		TDSetupS(Xn, A_angle, bx, by, bz, nu, p1, e13, Exx1Tn, Eyy1Tn, Ezz1Tn, Exy1Tn, Exz1Tn, Eyz1Tn);
		TDSetupS(Xn, B_angle, bx, by, bz, nu, p2, vectorOpposite(e12), Exx2Tn, Eyy2Tn, Ezz2Tn, Exy2Tn, Exz2Tn, Eyz2Tn);
		TDSetupS(Xn, C_angle, bx, by, bz, nu, p3, vectorOpposite(e23), Exx3Tn, Eyy3Tn, Ezz3Tn, Exy3Tn, Exz3Tn, Eyz3Tn);
	}
	//printf("%f %f %f\n",X1[0][0],X1[0][1],X1[0][2]);
	//printf("%.10f %.10f %.10f %.10f %.10f %.10f\n",Exx2Tp[0], Eyy2Tp[0], Ezz2Tp[0], Exy2Tp[0], Exz2Tp[0], Eyz2Tp[0]);
	


	std::vector<double> exx(Trimode.size(), 0.0);
	std::vector<double> eyy(Trimode.size(), 0.0);
	std::vector<double> ezz(Trimode.size(), 0.0);
	std::vector<double> exy(Trimode.size(), 0.0);
	std::vector<double> exz(Trimode.size(), 0.0);
	std::vector<double> eyz(Trimode.size(), 0.0);

	// casepLog
	for (size_t i = 0, p = 0; i < Trimode.size(); ++i) {
		if (Trimode[i] == 1) {
			exx[i] = Exx1Tp[p] + Exx2Tp[p] + Exx3Tp[p];
			eyy[i] = Eyy1Tp[p] + Eyy2Tp[p] + Eyy3Tp[p];
			ezz[i] = Ezz1Tp[p] + Ezz2Tp[p] + Ezz3Tp[p];
			exy[i] = Exy1Tp[p] + Exy2Tp[p] + Exy3Tp[p];
			exz[i] = Exz1Tp[p] + Exz2Tp[p] + Exz3Tp[p];
			eyz[i] = Eyz1Tp[p] + Eyz2Tp[p] + Eyz3Tp[p];
			++p;
		}
	}

	// casenLog
	for (size_t i = 0, n = 0; i < Trimode.size(); ++i) {
		if (Trimode[i] == -1) {
			exx[i] = Exx1Tn[n] + Exx2Tn[n] + Exx3Tn[n];
			eyy[i] = Eyy1Tn[n] + Eyy2Tn[n] + Eyy3Tn[n];
			ezz[i] = Ezz1Tn[n] + Ezz2Tn[n] + Ezz3Tn[n];
			exy[i] = Exy1Tn[n] + Exy2Tn[n] + Exy3Tn[n];
			exz[i] = Exz1Tn[n] + Exz2Tn[n] + Exz3Tn[n];
			eyz[i] = Eyz1Tn[n] + Eyz2Tn[n] + Eyz3Tn[n];
			++n;
		}
	}

	

	// TensTrans
	// void TensTrans(const std::vector<double>& exx, ... double Vnorm[3], ...);
	//std::vector<double> Exx, Eyy, Ezz, Exy, Exz, Eyz;
	//double A[3][3] = {
    //{ Vnorm[0], Vstrike[0], Vdip[0] },
    //{ Vnorm[1], Vstrike[1], Vdip[1] },
    //{ Vnorm[2], Vstrike[2], Vdip[2] }
	//};
	double A[3][3] = {
    { Vnorm[0], Vnorm[1], Vnorm[2] },
    { Vstrike[0], Vstrike[1], Vstrike[2] },
    { Vdip[0], Vdip[1], Vdip[2] }
	};

	TensTrans(exx, eyy, ezz, exy, exz, eyz, A, exx.size());
	//printf("%.20f %.20f %.20f %.20f %.20f %.20f\n",exx[0], eyy[0],ezz[0], exy[0], exz[0], eyz[0]);
	

    Sxx.resize(n);
    Syy.resize(n);
    Szz.resize(n);
    Sxy.resize(n);
    Sxz.resize(n);
    Syz.resize(n);

    for (size_t i = 0; i < n; ++i) {
        double trace = exx[i] + eyy[i] + ezz[i];
        Sxx[i] = 2 * mu * exx[i] + lambda_ * trace;
        Syy[i] = 2 * mu * eyy[i] + lambda_ * trace;
        Szz[i] = 2 * mu * ezz[i] + lambda_ * trace;
        Sxy[i] = 2 * mu * exy[i];
        Sxz[i] = 2 * mu * exz[i];
        Syz[i] = 2 * mu * eyz[i];
        Exx[i]=exx[i];
        Eyy[i]=eyy[i];
        Ezz[i]=ezz[i];
        Exy[i]=exy[i];
        Exz[i]=exz[i];
        Eyz[i]=eyz[i];
    }
}


void TDstressEachSourceAtReceiver(
    double x, double y, double z,
    const std::vector<std::vector<double>>& P1_list,
    const std::vector<std::vector<double>>& P2_list,
    const std::vector<std::vector<double>>& P3_list,
    double Ss, double Ds, double Ts,
    double mu, double lambda_,
    std::vector<std::vector<double>>& StressList,
    std::vector<std::vector<double>>& StrainList)
{
    size_t N = P1_list.size();

    StressList.clear();
    StrainList.clear();
    StressList.reserve(N);
    StrainList.reserve(N);

    std::vector<double> X(1, x), Y(1, y), Z(1, z);
    std::vector<double> sxx(1), syy(1), szz(1), sxy(1), sxz(1), syz(1);
    std::vector<double> exx(1), eyy(1), ezz(1), exy(1), exz(1), eyz(1);

    for (size_t i = 0; i < N; ++i) {
        TDstressFS(X, Y, Z,
                   P1_list[i], P2_list[i], P3_list[i],
                   Ss, Ds, Ts,
                   mu, lambda_,
                   sxx, syy, szz, sxy, sxz, syz,
                   exx, eyy, ezz, exy, exz, eyz);

        StressList.push_back({sxx[0], syy[0], szz[0], sxy[0], sxz[0], syz[0]});
        StrainList.push_back({exx[0], eyy[0], ezz[0], exy[0], exz[0], eyz[0]});
    }
}



/*
void TDstressFS(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z,
                const std::vector<double>& P1, const std::vector<double>& P2, const std::vector<double>& P3,
                double Ss, double Ds, double Ts, double mu, double lambda_,
                std::vector<double>& Sxx, std::vector<double>& Syy, std::vector<double>& Szz,
                std::vector<double>& Sxy, std::vector<double>& Sxz, std::vector<double>& Syz,
                std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
                std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz);*/


extern "C" void TDstressFS_C1(
    const double* X, const double* Y, const double* Z,
	size_t n,
    const double* P1, const double* P2, const double* P3,
    double Ss, double Ds, double Ts,
    double mu, double lambda_,
    
    double* Sxx, double* Syy, double* Szz,
    double* Sxy, double* Sxz, double* Syz,
    double* Exx, double* Eyy, double* Ezz,
    double* Exy, double* Exz, double* Eyz)
{
	//printf("n=%d",n);
    // 
    std::vector<double> Xv(X, X + n);
    std::vector<double> Yv(Y, Y + n);
    std::vector<double> Zv(Z, Z + n);

    // P1, P2, P3 
    std::vector<double> P1v(P1, P1 + 3);
    std::vector<double> P2v(P2, P2 + 3);
    std::vector<double> P3v(P3, P3 + 3);

    std::vector<double> Sxxv(n), Syyv(n), Szzv(n);
    std::vector<double> Sxyv(n), Sxzv(n), Syzv(n);
    std::vector<double> Exxv(n), Eyyv(n), Ezzv(n);
    std::vector<double> Exyv(n), Exzv(n), Eyzv(n);


    TDstressFS(Xv, Yv, Zv, P1v, P2v, P3v, Ss, Ds, Ts, mu, lambda_,
               Sxxv, Syyv, Szzv, Sxyv, Sxzv, Syzv,
               Exxv, Eyyv, Ezzv, Exyv, Exzv, Eyzv);

    for (int i = 0; i < n; ++i) {
        Sxx[i] = Sxxv[i]; Syy[i] = Syyv[i]; Szz[i] = Szzv[i];
        Sxy[i] = Sxyv[i]; Sxz[i] = Sxzv[i]; Syz[i] = Syzv[i];
        Exx[i] = Exxv[i]; Eyy[i] = Eyyv[i]; Ezz[i] = Ezzv[i];
        Exy[i] = Exyv[i]; Exz[i] = Exzv[i]; Eyz[i] = Eyzv[i];
    }
}


// Computes free surface correction strain partials over arrays of points
void AngDisStrainFSC(
    const vector<double>& y1,
    const vector<double>& y2,
    const vector<double>& y3,
    double beta,
    double b1, double b2, double b3,
    double nu,
    double a,
    vector<double>& v11,
    vector<double>& v22,
    vector<double>& v33,
    vector<double>& v12,
    vector<double>& v13,
    vector<double>& v23)
{
    size_t N = y1.size();
    // vector<double> rFib_ry1(N);
    // vector<double> rFib_ry2(N);
    // vector<double> rFib_ry3(N);

    v11.resize(N);
    v22.resize(N);
    v33.resize(N);
    v12.resize(N);
    v13.resize(N);
    v23.resize(N);
    // Precompute trigonometric values
    double sinB = sin(beta);
    double cosB = cos(beta);
    double cotB = 1.0 / tan(beta);
    double N1 = 1.0 - 2.0 * nu;

    for (size_t i = 0; i < N; ++i) {
        // Image coordinates
        double y3b = y3[i] + 2.0 * a;
        double z1b = y1[i] * cosB + y3b * sinB;
        double z3b = -y1[i] * sinB + y3b * cosB;

        double y1_sq = y1[i] * y1[i];
        double y2_sq = y2[i] * y2[i];
        double rb2 = y1_sq + y2_sq + y3b * y3b;
        double rb = sqrt(rb2);
        if (rb < 1e-12) {
            v11[i] = 0.0;
            continue;
        }

        double rb3 = rb2 * rb;
        double rb5 = rb2 * rb2 * rb;
        double rb4 = rb2 * rb2;

        double W1 = rb * cosB + y3b;
        double W2 = cosB + a / rb;
        double W3 = cosB + y3b / rb;
        double W4 = nu + a / rb;
        double W5 = 2.0 * nu + a / rb;
        double W6 = rb + y3b;
        double W7 = rb + z3b;
        double W8 = y3[i] + a;
        double W9 = 1.0 + a / (rb * cosB);

        //cout<<"rb2 "<<rb2<<endl;
        // cout<<"cosB "<<cosB<<endl;
        // cout<<"a "<<a<<endl;
        // cout<<"rb "<<rb<<endl;
        //cout<<"W  "<<W1<<"\t"<<W2<<"\t"<<W3<<"\t"<<W4<<"\t"<<W5<<"\t"<<W6<<"\t"<<W7<<"\t"<<W8<<"\t"<<W9<<endl;
        //printf("%.10f\n",W6);
        // Partial derivatives
        double rFib_ry2 = z1b / (rb * (rb + z3b)) - y1[i] / (rb * (rb + y3b));
        double rFib_ry1 = y2[i] / (rb * (rb + y3b)) - cosB * y2[i] / (rb * (rb + z3b));
        double rFib_ry3 = -sinB * y2[i] / (rb * (rb + z3b));
        //cout<<"y1[i] z3b "<<y1[i]<<" "<<z3b<<endl;
        //cout<<"rFib_ry "<< rFib_ry1<<" "<<rFib_ry2<<"  "<<rFib_ry3<<endl;
        // === b1 term ===
        //cout<<"b1  "<<b1<<endl;
        //cout<<"b2  "<<b2<<endl;
        //cout<<"b3  "<<b3<<endl;
        v11[i] =b1 * (1.0 / 4.0 * (
            (-2.0 + 2.0 * nu) * N1 * rFib_ry1 * cotB * cotB
            - N1 * y2[i] / (W6 * W6) * ((1.0 - W5) * cotB - y1[i] / W6 * W4) / rb * y1[i]
            + N1 * y2[i] / W6 * (
                a / rb3 * y1[i] * cotB
                - 1.0 / W6 * W4
                + y1_sq / (W6 * W6) * W4 / rb
                + y1_sq / W6 * a / rb3
            )
            - N1 * y2[i] * cosB * cotB / (W7 * W7) * W2 * (y1[i] / rb - sinB)
            - N1 * y2[i] * cosB * cotB / W7 * a / rb3 * y1[i]
            - 3.0 * a * y2[i] * W8 * cotB / rb5 * y1[i]
            - y2[i] * W8 / (rb3 * W6) * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2) * y1[i]
            - y2[i] * W8 / (rb2 * W6 * W6) * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2) * y1[i]
            + y2[i] * W8 / (rb * W6) * (
                1.0 / W6 * W5
                - y1_sq / (W6 * W6) * W5 / rb
                - y1_sq / W6 * a / rb3
                + a / rb2
                - 2.0 * a * y1_sq / rb4
            )
            - y2[i] * W8 / (rb3 * W7) * (
                cosB / W7 * (
                    W1 * (N1 * cosB - a / rb) * cotB +
                    (2.0 - 2.0 * nu) * (rb * sinB - y1[i]) * cosB
                ) - a * y3b * cosB * cotB / rb2
            ) * y1[i]
            - y2[i] * W8 / (rb * W7 * W7) * (
                cosB / W7 * (
                    W1 * (N1 * cosB - a / rb) * cotB +
                    (2.0 - 2.0 * nu) * (rb * sinB - y1[i]) * cosB
                ) - a * y3b * cosB * cotB / rb2
            ) * (y1[i] / rb - sinB)
            + y2[i] * W8 / (rb * W7) * (
                -cosB / (W7 * W7) * (
                    W1 * (N1 * cosB - a / rb) * cotB +
                    (2.0 - 2.0 * nu) * (rb * sinB - y1[i]) * cosB
                ) * (y1[i] / rb - sinB)
                + cosB / W7 * (
                    1.0 / rb * cosB * y1[i] * (N1 * cosB - a / rb) * cotB
                    + W1 * a / rb3 * y1[i] * cotB
                    + (2.0 - 2.0 * nu) * (1.0 / rb * sinB * y1[i] - 1.0) * cosB
                )
                + 2.0 * a * y3b * cosB * cotB / rb4 * y1[i]
            )) / (PI * (1.0 - nu)))+
            b2 * (1. / 4 * (
                N1 * (((2 - 2 * nu) * cotB * cotB + nu) / rb * y1[i] / W6 -
                    ((2 - 2 * nu) * cotB * cotB + 1) * cosB * (y1[i] / rb - sinB) / W7)
                - N1 / (W6 * W6) * (-N1 * y1[i] * cotB + nu * y3b - a + a * y1[i] * cotB / rb + y1[i] * y1[i] / W6 * W4) / rb * y1[i]
                + N1 / W6 * (-N1 * cotB + a * cotB / rb - a * y1[i] * y1[i] * cotB / (rb * rb * rb)
                    + 2. * y1[i] / W6 * W4 - y1[i] * y1[i] * y1[i] / (W6 * W6) * W4 / rb
                    - y1[i] * y1[i] * y1[i] / W6 * a / (rb * rb * rb))
                + N1 * cotB / (W7 * W7) * (z1b * cosB - a * (rb * sinB - y1[i]) / rb / cosB) * (y1[i] / rb - sinB)
                - N1 * cotB / W7 * (cosB * cosB - a * (1. / rb * sinB * y1[i] - 1) / rb / cosB
                    + a * (rb * sinB - y1[i]) / (rb * rb * rb) / cosB * y1[i])
                - a * W8 * cotB / (rb * rb * rb) + 3 * a * y1[i] * y1[i] * W8 * cotB / (rb * rb * rb * rb * rb)
                - W8 / (W6 * W6) * (2 * nu + 1. / rb * (N1 * y1[i] * cotB + a) - y1[i] * y1[i] / rb / W6 * W5 - a * y1[i] * y1[i] / (rb * rb * rb)) / rb * y1[i]
                + W8 / W6 * (
                    -1. / (rb * rb * rb) * (N1 * y1[i] * cotB + a) * y1[i]
                    + 1. / rb * N1 * cotB
                    - 2. * y1[i] / rb / W6 * W5
                    + y1[i] * y1[i] * y1[i] / (rb * rb * rb) / W6 * W5
                    + y1[i] * y1[i] * y1[i] / (rb2) / (W6 * W6) * W5
                    + y1[i] * y1[i] * y1[i] / (rb2 * rb2) / W6 * a
                    - 2 * a / (rb * rb * rb) * y1[i]
                    + 3 * a * y1[i] * y1[i] * y1[i] / (rb * rb * rb * rb * rb))
                - W8 * cotB / (W7 * W7) * (-cosB * sinB
                    + a * y1[i] * y3b / (rb * rb * rb) / cosB
                    + (rb * sinB - y1[i]) / rb * ((2 - 2 * nu) * cosB - W1 / W7 * W9)) * (y1[i] / rb - sinB)
                + W8 * cotB / W7 * (
                    a * y3b / (rb * rb * rb) / cosB
                    - 3 * a * y1[i] * y1[i] * y3b / (rb * rb * rb * rb * rb) / cosB
                    + (1. / rb * sinB * y1[i] - 1) / rb * ((2 - 2 * nu) * cosB - W1 / W7 * W9)
                    - (rb * sinB - y1[i]) / (rb * rb * rb) * ((2 - 2 * nu) * cosB - W1 / W7 * W9) * y1[i]
                    + (rb * sinB - y1[i]) / rb * (-1. / rb * cosB * y1[i] / W7 * W9
                    + W1 / (W7 * W7) * W9 * (y1[i] / rb - sinB)
                    + W1 / W7 * a / (rb * rb * rb) / cosB * y1[i])
                )
            ) / PI / (1 - nu))+
            b3 * (1.0 / 4.0 * (
                N1 * (
                    - y2[i] / (W6 * W6) * (1 + a / rb) / rb * y1[i]
                    - y2[i] / W6 * a / (rb * rb * rb) * y1[i]
                    + y2[i] * cosB / (W7 * W7) * W2 * (y1[i] / rb - sinB)
                    + y2[i] * cosB / W7 * a / (rb * rb * rb) * y1[i]
                )
                + y2[i] * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) * y1[i]
                - y2[i] * W8 / rb * (-2 * a / (rb2 * rb2) * y1[i] - 1.0 / (W6 * W6) / rb * y1[i])
                - y2[i] * W8 * cosB / (rb * rb * rb) / W7 * (W1 / W7 * W2 + a * y3b / rb2) * y1[i]
                - y2[i] * W8 * cosB / rb / (W7 * W7) * (W1 / W7 * W2 + a * y3b / rb2) * (y1[i] / rb - sinB)
                + y2[i] * W8 * cosB / rb / W7 * (
                    1.0 / rb * cosB * y1[i] / W7 * W2
                    - W1 / (W7 * W7) * W2 * (y1[i] / rb - sinB)
                    - W1 / W7 * a / (rb * rb * rb) * y1[i]
                    - 2 * a * y3b / (rb2 * rb2) * y1[i]
                )) / PI / (1 - nu));



        
        //cout<<"v11 "<<v11[i]<<endl;
        v22[i] = b1*(1.0/4.0*(N1*(((2.0-2.0*nu)*cotB*cotB-nu)/rb*y2[i]/W6-((2.0-2.0*nu)*cotB*cotB+1.0-2.0*nu)*cosB/rb*y2[i]/W7)+N1/W6/W6*(y1[i]*cotB*(1.0-W5)+nu*y3b-a+y2[i]*y2[i]/W6*W4)/rb*y2[i]-N1/W6*(a*y1[i]*cotB/rb/rb/rb*y2[i]+2.0*y2[i]/W6*W4-y2[i]*y2[i]*y2[i]/W6/W6*W4/rb-y2[i]*y2[i]*y2[i]/W6*a/rb/rb/rb)+N1*z1b*cotB/W7/W7*W2/rb*y2[i]+N1*z1b*cotB/W7*a/rb/rb/rb*y2[i]+3.0*a*y2[i]*W8*cotB/rb/rb/rb/rb/rb*y1[i]-W8/W6/W6*(-2.0*nu+1.0/rb*(N1*y1[i]*cotB-a)+y2[i]*y2[i]/rb/W6*W5+a*y2[i]*y2[i]/rb/rb/rb)/rb*y2[i]+W8/W6*(-1.0/rb/rb/rb*(N1*y1[i]*cotB-a)*y2[i]+2.0*y2[i]/rb/W6*W5-y2[i]*y2[i]*y2[i]/rb/rb/rb/W6*W5-y2[i]*y2[i]*y2[i]/rb2/W6/W6*W5-y2[i]*y2[i]*y2[i]/rb2/rb2/W6*a+2.0*a/rb/rb/rb*y2[i]-3.0*a*y2[i]*y2[i]*y2[i]/rb/rb/rb/rb/rb)-W8/W7/W7*(cosB*cosB-1.0/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb/rb/rb-1.0/rb/W7*(y2[i]*y2[i]*cosB*cosB-a*z1b*cotB/rb*W1))/rb*y2[i]+W8/W7*(1.0/rb/rb/rb*(N1*z1b*cotB+a*cosB)*y2[i]-3.0*a*y3b*z1b*cotB/rb/rb/rb/rb/rb*y2[i]+1.0/rb/rb/rb/W7*(y2[i]*y2[i]*cosB*cosB-a*z1b*cotB/rb*W1)*y2[i]+1.0/rb2/W7/W7*(y2[i]*y2[i]*cosB*cosB-a*z1b*cotB/rb*W1)*y2[i]-1.0/rb/W7*(2.0*y2[i]*cosB*cosB+a*z1b*cotB/rb/rb/rb*W1*y2[i]-a*z1b*cotB/rb2*cosB*y2[i])))/PI/(1.0-nu))+
        b2*(1.0/4.0*((2.0-2.0*nu)*N1*rFib_ry2*cotB*cotB+N1/W6*((W5-1)*cotB+y1[i]/W6*W4)-N1*y2[i]*y2[i]/W6/W6*((W5-1)*cotB+y1[i]/W6*W4)/rb+N1*y2[i]/W6*(-a/rb/rb/rb*y2[i]*cotB-y1[i]/W6/W6*W4/rb*y2[i]-y2[i]/W6*a/rb/rb/rb*y1[i])-N1*cotB/W7*W9+N1*y2[i]*y2[i]*cotB/W7/W7*W9/rb+N1*y2[i]*y2[i]*cotB/W7*a/rb/rb/rb/cosB-a*W8*cotB/rb/rb/rb+3.0*a*y2[i]*y2[i]*W8*cotB/rb/rb/rb/rb/rb+W8/rb/W6*(N1*cotB-2.0*nu*y1[i]/W6-a*y1[i]/rb*(1.0/rb+1.0/W6))-y2[i]*y2[i]*W8/rb/rb/rb/W6*(N1*cotB-2.0*nu*y1[i]/W6-a*y1[i]/rb*(1.0/rb+1.0/W6))-y2[i]*y2[i]*W8/rb2/W6/W6*(N1*cotB-2.0*nu*y1[i]/W6-a*y1[i]/rb*(1.0/rb+1.0/W6))+y2[i]*W8/rb/W6*(2.0*nu*y1[i]/W6/W6/rb*y2[i]+a*y1[i]/rb/rb/rb*(1.0/rb+1.0/W6)*y2[i]-a*y1[i]/rb*(-1.0/rb/rb/rb*y2[i]-1.0/W6/W6/rb*y2[i]))+W8*cotB/rb/W7*((-2.0+2.0*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)-y2[i]*y2[i]*W8*cotB/rb/rb/rb/W7*((-2.0+2.0*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)-y2[i]*y2[i]*W8*cotB/rb2/W7/W7*((-2.0+2.0*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)+y2[i]*W8*cotB/rb/W7*(1.0/rb*cosB*y2[i]/W7*W9-W1/W7/W7*W9/rb*y2[i]-W1/W7*a/rb/rb/rb/cosB*y2[i]-2.0*a*y3b/rb2/rb2/cosB*y2[i]))/PI/(1.0-nu))+
        b3*(1.0/4.0*(N1*(-sinB/rb*y2[i]/W7+y2[i]/W6/W6*(1.0+a/rb)/rb*y1[i]+y2[i]/W6*a/rb/rb/rb*y1[i]-z1b/W7/W7*W2/rb*y2[i]-z1b/W7*a/rb/rb/rb*y2[i])-y2[i]*W8/rb/rb/rb*(a/rb2+1.0/W6)*y1[i]+y1[i]*W8/rb*(-2.0*a/rb2/rb2*y2[i]-1.0/W6/W6/rb*y2[i])+W8/W7/W7*(sinB*(cosB-a/rb)+z1b/rb*(1.0+a*y3b/rb2)-1.0/rb/W7*(y2[i]*y2[i]*cosB*sinB-a*z1b/rb*W1))/rb*y2[i]-W8/W7*(sinB*a/rb/rb/rb*y2[i]-z1b/rb/rb/rb*(1.0+a*y3b/rb2)*y2[i]-2.0*z1b/rb/rb/rb/rb/rb*a*y3b*y2[i]+1.0/rb/rb/rb/W7*(y2[i]*y2[i]*cosB*sinB-a*z1b/rb*W1)*y2[i]+1.0/rb2/W7/W7*(y2[i]*y2[i]*cosB*sinB-a*z1b/rb*W1)*y2[i]-1.0/rb/W7*(2.0*y2[i]*cosB*sinB+a*z1b/rb/rb/rb*W1*y2[i]-a*z1b/rb2*cosB*y2[i])))/PI/(1.0-nu)); 
            
        //v33[i] = b1 * (1.0 / 4.0 * ((2.0 - 2.0 * nu) * (N1 * rFib_ry3 * cotB - y2[i] / pow(W6, 2.0) * W5 * (y3[i] / rb + 1.0) - 1.0 / 2.0 * y2[i] / W6 * a / pow(rb, 3.0) * 2.0 * y3[i] + y2[i] * cosB / pow(W7, 2.0) * W2 * W3 + 1.0 / 2.0 * y2[i] * cosB / W7 * a / pow(rb, 3.0) * 2.0 * y3[i]) + y2[i] / rb * (2.0 * nu / W6 + a / rb2) - 1.0 / 2.0 * y2[i] * W8 / pow(rb, 3.0) * (2.0 * nu / W6 + a / rb2) * 2.0 * y3[i] + y2[i] * W8 / rb * (-2.0 * nu / pow(W6, 2.0) * (y3[i] / rb + 1.0) - a / pow(rb2, 2.0) * 2.0 * y3[i]) + y2[i] * cosB / rb / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) - 1.0 / 2.0 * y2[i] * W8 * cosB / pow(rb, 3.0) / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) * 2.0 * y3[i] - y2[i] * W8 * cosB / rb / pow(W7, 2.0) * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) * W3 + y2[i] * W8 * cosB / rb / W7 * (-(cosB * y3[i] / rb + 1.0) / W7 * W2 + W1 / pow(W7, 2.0) * W2 * W3 + 1.0 / 2.0 * W1 / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] - a / rb2 + a * y3[i] / pow(rb2, 2.0) * 2.0 * y3[i])) / PI / (1.0 - nu));
        //b2 * (1.0 / 4.0 * ((-2.0 + 2.0 * nu) * N1 * cotB * ((y3[i] / rb + 1.0) / W6 - cosB * W3 / W7) + (2.0 - 2.0 * nu) * y1[i] / pow(W6, 2.0) * W5 * (y3[i] / rb + 1.0) + 1.0 / 2.0 * (2.0 - 2.0 * nu) * y1[i] / W6 * a / pow(rb, 3.0) * 2.0 * y3[i] + (2.0 - 2.0 * nu) * sinB / W7 * W2 - (2.0 - 2.0 * nu) * z1b / pow(W7, 2.0) * W2 * W3 - 1.0 / 2.0 * (2.0 - 2.0 * nu) * z1b / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] + 1.0 / rb * (N1 * cotB - 2.0 * nu * y1[i] / W6 - a * y1[i] / rb2) - 1.0 / 2.0 * W8 / pow(rb, 3.0) * (N1 * cotB - 2.0 * nu * y1[i] / W6 - a * y1[i] / rb2) * 2.0 * y3[i] + W8 / rb * (2.0 * nu * y1[i] / pow(W6, 2.0) * (y3[i] / rb + 1.0) + a * y1[i] / pow(rb2, 2.0) * 2.0 * y3[i]) - 1.0 / W7 * (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) + a / rb * (sinB - y3[i] * z1b / rb2 - z1b * W1 / rb / W7)) + W8 / pow(W7, 2.0) * (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) + a / rb * (sinB - y3[i] * z1b / rb2 - z1b * W1 / rb / W7)) * W3 - W8 / W7 * ((cosB * y3[i] / rb + 1.0) * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) - 1.0 / 2.0 * W1 * cotB / pow(rb, 3.0) * ((2.0 - 2.0 * nu) * cosB - W1 / W7) * 2.0 * y3[i] + W1 * cotB / rb * (-(cosB * y3[i] / rb + 1.0) / W7 + W1 / pow(W7, 2.0) * W3) - 1.0 / 2.0 * a / pow(rb, 3.0) * (sinB - y3[i] * z1b / rb2 - z1b * W1 / rb / W7) * 2.0 * y3[i] + a / rb * (-z1b / rb2 - y3[i] * sinB / rb2 + y3[i] * z1b / pow(rb2, 2.0) * 2.0 * y3[i] - sinB * W1 / rb / W7 - z1b * (cosB * y3[i] / rb + 1.0) / rb / W7 + 1.0 / 2.0 * z1b * W1 / pow(rb, 3.0) / W7 * 2.0 * y3[i] + z1b * W1 / rb / pow(W7, 2.0) * W3))) / PI / (1.0 - nu)) + 
        //b3 * (1.0 / 4.0 * ((2.0 - 2.0 * nu) * rFib_ry3 - (2.0 - 2.0 * nu) * y2[i] * sinB / pow(W7, 2.0) * W2 * W3 - 1.0 / 2.0 * (2.0 - 2.0 * nu) * y2[i] * sinB / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] + y2[i] * sinB / rb / W7 * (1.0 + W1 / W7 * W2 + a * y3[i] / rb2) - 1.0 / 2.0 * y2[i] * W8 * sinB / pow(rb, 3.0) / W7 * (1.0 + W1 / W7 * W2 + a * y3[i] / rb2) * 2.0 * y3[i] - y2[i] * W8 * sinB / rb / pow(W7, 2.0) * (1.0 + W1 / W7 * W2 + a * y3[i] / rb2) * W3 + y2[i] * W8 * sinB / rb / W7 * ((cosB * y3[i] / rb + 1.0) / W7 * W2 - W1 / pow(W7, 2.0) * W2 * W3 - 1.0 / 2.0 * W1 / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] + a / rb2 - a * y3[i] / pow(rb2, 2.0) * 2.0 * y3[i])) / PI / (1.0 - nu));
        //v33[i] = b1 * (1.0 / 4.0 * ((2.0 - 2.0 * nu) * (N1 * rFib_ry3 * cotB - y2[i] / pow(W6, 2.0) * W5 * (y3[i] / rb + 1.0) - 1.0 / 2.0 * y2[i] / W6 * a / pow(rb, 3.0) * 2.0 * y3[i] + y2[i] * cosB / pow(W7, 2.0) * W2 * W3 + 1.0 / 2.0 * y2[i] * cosB / W7 * a / pow(rb, 3.0) * 2.0 * y3[i]) + y2[i] / rb * (2.0 * nu / W6 + a / rb2) - 1.0 / 2.0 * y2[i] * W8 / pow(rb, 3.0) * (2.0 * nu / W6 + a / rb2) * 2.0 * y3[i] + y2[i] * W8 / rb * (-2.0 * nu / pow(W6, 2.0) * (y3[i] / rb + 1.0) - a / pow(rb2, 2.0) * 2.0 * y3[i]) + y2[i] * cosB / rb / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) - 1.0 / 2.0 * y2[i] * W8 * cosB / pow(rb, 3.0) / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) * 2.0 * y3[i] - y2[i] * W8 * cosB / rb / pow(W7, 2.0) * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) * W3 + y2[i] * W8 * cosB / rb / W7 * (-(cosB * y3[i] / rb + 1.0) / W7 * W2 + W1 / pow(W7, 2.0) * W2 * W3 + 1.0 / 2.0 * W1 / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] - a / rb2 + a * y3[i] / pow(rb2, 2.0) * 2.0 * y3[i])) / PI / (1.0 - nu)) + b2 * (1.0 / 4.0 * ((-2.0 + 2.0 * nu) * N1 * cotB * ((y3[i] / rb + 1.0) / W6 - cosB * W3 / W7) + (2.0 - 2.0 * nu) * y1[i] / pow(W6, 2.0) * W5 * (y3[i] / rb + 1.0) + 1.0 / 2.0 * (2.0 - 2.0 * nu) * y1[i] / W6 * a / pow(rb, 3.0) * 2.0 * y3[i] + (2.0 - 2.0 * nu) * sinB / W7 * W2 - (2.0 - 2.0 * nu) * z1b / pow(W7, 2.0) * W2 * W3 - 1.0 / 2.0 * (2.0 - 2.0 * nu) * z1b / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] + 1.0 / rb * (N1 * cotB - 2.0 * nu * y1[i] / W6 - a * y1[i] / rb2) - 1.0 / 2.0 * W8 / pow(rb, 3.0) * (N1 * cotB - 2.0 * nu * y1[i] / W6 - a * y1[i] / rb2) * 2.0 * y3[i] + W8 / rb * (2.0 * nu * y1[i] / pow(W6, 2.0) * (y3[i] / rb + 1.0) + a * y1[i] / pow(rb2, 2.0) * 2.0 * y3[i]) - 1.0 / W7 * (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) + a / rb * (sinB - y3[i] * z1b / rb2 - z1b * W1 / rb / W7)) + W8 / pow(W7, 2.0) * (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) + a / rb * (sinB - y3[i] * z1b / rb2 - z1b * W1 / rb / W7)) * W3 - W8 / W7 * ((cosB * y3[i] / rb + 1.0) * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) - 1.0 / 2.0 * W1 * cotB / pow(rb, 3.0) * ((2.0 - 2.0 * nu) * cosB - W1 / W7) * 2.0 * y3[i] + W1 * cotB / rb * (-(cosB * y3[i] / rb + 1.0) / W7 + W1 / pow(W7, 2.0) * W3) - 1.0 / 2.0 * a / pow(rb, 3.0) * (sinB - y3[i] * z1b / rb2 - z1b * W1 / rb / W7) * 2.0 * y3[i] + a / rb * (-z1b / rb2 - y3[i] * sinB / rb2 + y3[i] * z1b / pow(rb2, 2.0) * 2.0 * y3[i] - sinB * W1 / rb / W7 - z1b * (cosB * y3[i] / rb + 1.0) / rb / W7 + 1.0 / 2.0 * z1b * W1 / pow(rb, 3.0) / W7 * 2.0 * y3[i] + z1b * W1 / rb / pow(W7, 2.0) * W3))) / PI / (1.0 - nu)) + b3 * (1.0 / 4.0 * ((2.0 - 2.0 * nu) * rFib_ry3 - (2.0 - 2.0 * nu) * y2[i] * sinB / pow(W7, 2.0) * W2 * W3 - 1.0 / 2.0 * (2.0 - 2.0 * nu) * y2[i] * sinB / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] + y2[i] * sinB / rb / W7 * (1.0 + W1 / W7 * W2 + a * y3[i] / rb2) - 1.0 / 2.0 * y2[i] * W8 * sinB / pow(rb, 3.0) / W7 * (1.0 + W1 / W7 * W2 + a * y3[i] / rb2) * 2.0 * y3[i] - y2[i] * W8 * sinB / rb / pow(W7, 2.0) * (1.0 + W1 / W7 * W2 + a * y3[i] / rb2) * W3 + y2[i] * W8 * sinB / rb / W7 * ((cosB * y3[i] / rb + 1.0) / W7 * W2 - W1 / pow(W7, 2.0) * W2 * W3 - 1.0 / 2.0 * W1 / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] + a / rb2 - a * y3[i] / pow(rb2, 2.0) * 2.0 * y3[i])) / PI / (1.0 - nu));
        //v33[i]=b1 * (1.0 / 4.0 * ((2.0 - 2.0 * nu) * (N1 * rFib_ry3 * cotB - y2[i] / pow(W6, 2.0) * W5 * (y3[i] / rb + 1.0) - 1.0 / 2.0 * y2[i] / W6 * a / pow(rb, 3.0) * 2.0 * y3[i] + y2[i] * cosB / pow(W7, 2.0) * W2 * W3 + 1.0 / 2.0 * y2[i] * cosB / W7 * a / pow(rb, 3.0) * 2.0 * y3[i]) + y2[i] / rb * (2.0 * nu / W6 + a / rb2) - 1.0 / 2.0 * y2[i] * W8 / pow(rb, 3.0) * (2.0 * nu / W6 + a / rb2) * 2.0 * y3[i] + y2[i] * W8 / rb * (-2.0 * nu / pow(W6, 2.0) * (y3[i] / rb + 1.0) - a / pow(rb2, 2.0) * 2.0 * y3[i]) + y2[i] * cosB / rb / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) - 1.0 / 2.0 * y2[i] * W8 * cosB / pow(rb, 3.0) / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) * 2.0 * y3[i] - y2[i] * W8 * cosB / rb / pow(W7, 2.0) * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3[i] / rb2) * W3 + y2[i] * W8 * cosB / rb / W7 * (-(cosB * y3[i] / rb + 1.0) / W7 * W2 + W1 / pow(W7, 2.0) * W2 * W3 + 1.0 / 2.0 * W1 / W7 * a / pow(rb, 3.0) * 2.0 * y3[i] - a / rb2 + a * y3[i] / pow(rb2, 2.0) * 2.0 * y3[i])) / PI / (1.0 - nu));
        v33[i] =b1 * (1.0 / 4.0 * (
            (2 - 2 * nu) * (
                N1 * rFib_ry3 * cotB
                - y2[i] / (W6 * W6) * W5 * (y3b / rb + 1)
                - 0.5 * y2[i] / W6 * a / (rb * rb * rb) * 2.0 * y3b
                + y2[i] * cosB / (W7 * W7) * W2 * W3
                + 0.5 * y2[i] * cosB / W7 * a / (rb * rb * rb) * 2.0 * y3b
            )
            + y2[i] / rb * (2 * nu / W6 + a / rb2)
            - 0.5 * y2[i] * W8 / (rb * rb * rb) * (2 * nu / W6 + a / rb2) * 2.0 * y3b
            + y2[i] * W8 / rb * (-2 * nu / (W6 * W6) * (y3b / rb + 1) - a / (rb2 * rb2) * 2.0 * y3b)
            + y2[i] * cosB / rb / W7 * (1 - 2 * nu - W1 / W7 * W2 - a * y3b / rb2)
            - 0.5 * y2[i] * W8 * cosB / (rb * rb * rb) / W7 * (1 - 2 * nu - W1 / W7 * W2 - a * y3b / rb2) * 2.0 * y3b
            - y2[i] * W8 * cosB / rb / (W7 * W7) * (1 - 2 * nu - W1 / W7 * W2 - a * y3b / rb2) * W3
            + y2[i] * W8 * cosB / rb / W7 * (
                -(cosB * y3b / rb + 1) / W7 * W2
                + W1 / (W7 * W7) * W2 * W3
                + 0.5 * W1 / W7 * a / (rb * rb * rb) * 2.0 * y3b
                - a / rb2
                + a * y3b / (rb2 * rb2) * 2.0 * y3b
            )
        ) / PI / (1 - nu))+
        b2 * (1.0 / 4.0 * (
            (-2 + 2 * nu) * N1 * cotB * ((y3b / rb + 1.0) / W6 - cosB * W3 / W7)
            + (2 - 2 * nu) * y1[i] / (W6 * W6) * W5 * (y3b / rb + 1.0)
            + 0.5 * (2 - 2 * nu) * y1[i] / W6 * a / (rb * rb * rb) * 2.0 * y3b
            + (2 - 2 * nu) * sinB / W7 * W2
            - (2 - 2 * nu) * z1b / (W7 * W7) * W2 * W3
            - 0.5 * (2 - 2 * nu) * z1b / W7 * a / (rb * rb * rb) * 2.0 * y3b
            + 1.0 / rb * (N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb2)
            - 0.5 * W8 / (rb * rb * rb) * (N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb2) * 2.0 * y3b
            + W8 / rb * (
                2 * nu * y1[i] / (W6 * W6) * (y3b / rb + 1.0)
                + a * y1[i] / (rb2 * rb2) * 2.0 * y3b
            )
            - 1.0 / W7 * (
                cosB * sinB
                + W1 * cotB / rb * ((2 - 2 * nu) * cosB - W1 / W7)
                + a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)
            )
            + W8 / (W7 * W7) * (
                cosB * sinB
                + W1 * cotB / rb * ((2 - 2 * nu) * cosB - W1 / W7)
                + a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)
            ) * W3
            - W8 / W7 * (
                (cosB * y3b / rb + 1.0) * cotB / rb * ((2 - 2 * nu) * cosB - W1 / W7)
                - 0.5 * W1 * cotB / (rb * rb * rb) * ((2 - 2 * nu) * cosB - W1 / W7) * 2.0 * y3b
                + W1 * cotB / rb * (-(cosB * y3b / rb + 1.0) / W7 + W1 / (W7 * W7) * W3)
                - 0.5 * a / (rb * rb * rb) * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7) * 2.0 * y3b
                + a / rb * (
                    -z1b / rb2
                    - y3b * sinB / rb2
                    + y3b * z1b / (rb2 * rb2) * 2.0 * y3b
                    - sinB * W1 / rb / W7
                    - z1b * (cosB * y3b / rb + 1.0) / rb / W7
                    + 0.5 * z1b * W1 / (rb * rb * rb) / W7 * 2.0 * y3b
                    + z1b * W1 / rb / (W7 * W7) * W3
                )
            )
        ) / PI / (1 - nu))+
        b3 * (1.0 / 4.0 * (
            (2 - 2 * nu) * rFib_ry3
            - (2 - 2 * nu) * y2[i] * sinB / (W7 * W7) * W2 * W3
            - 0.5 * (2 - 2 * nu) * y2[i] * sinB / W7 * a / (rb * rb * rb) * 2.0 * y3b
            + y2[i] * sinB / rb / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2)
            - 0.5 * y2[i] * W8 * sinB / (rb * rb * rb) / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2) * 2.0 * y3b
            - y2[i] * W8 * sinB / rb / (W7 * W7) * (1.0 + W1 / W7 * W2 + a * y3b / rb2) * W3
            + y2[i] * W8 * sinB / rb / W7 * (
                (cosB * y3b / rb + 1.0) / W7 * W2
                - W1 / (W7 * W7) * W2 * W3
                - 0.5 * W1 / W7 * a / (rb * rb * rb) * 2.0 * y3b
                + a / rb2
                - a * y3b / (rb2 * rb2) * 2.0 * y3b
            )
        ) / PI / (1 - nu));
        
        v12[i] =
        b1 / 2.0 * (1.0 / 4.0 * (
            (-2 + 2 * nu) * N1 * rFib_ry2 * cotB * cotB
            + N1 / W6 * ((1 - W5) * cotB - y1[i] / W6 * W4)
            - N1 * pow(y2[i], 2) / (W6 * W6) * ((1 - W5) * cotB - y1[i] / W6 * W4) / rb
            + N1 * y2[i] / W6 * (
                a / pow(rb, 3) * y2[i] * cotB
                + y1[i] / (W6 * W6) * W4 / rb * y2[i]
                + y2[i] / W6 * a / pow(rb, 3) * y1[i]
            )
            + N1 * cosB * cotB / W7 * W2
            - N1 * pow(y2[i], 2) * cosB * cotB / (W7 * W7) * W2 / rb
            - N1 * pow(y2[i], 2) * cosB * cotB / W7 * a / pow(rb, 3)
            + a * W8 * cotB / pow(rb, 3)
            - 3 * a * pow(y2[i], 2) * W8 * cotB / pow(rb, 5)
            + W8 / rb / W6 * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2)
            - pow(y2[i], 2) * W8 / pow(rb, 3) / W6 * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2)
            - pow(y2[i], 2) * W8 / rb2 / (W6 * W6) * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2)
            + y2[i] * W8 / rb / W6 * (
                - y1[i] / (W6 * W6) * W5 / rb * y2[i]
                - y2[i] / W6 * a / pow(rb, 3) * y1[i]
                - 2 * a * y1[i] / pow(rb2, 2) * y2[i]
            )
            + W8 / rb / W7 * (
                cosB / W7 * (
                    W1 * (N1 * cosB - a / rb) * cotB
                    + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                )
                - a * y3b * cosB * cotB / rb2
            )
            - pow(y2[i], 2) * W8 / pow(rb, 3) / W7 * (
                cosB / W7 * (
                    W1 * (N1 * cosB - a / rb) * cotB
                    + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                )
                - a * y3b * cosB * cotB / rb2
            )
            - pow(y2[i], 2) * W8 / rb2 / (W7 * W7) * (
                cosB / W7 * (
                    W1 * (N1 * cosB - a / rb) * cotB
                    + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                )
                - a * y3b * cosB * cotB / rb2
            )
            + y2[i] * W8 / rb / W7 * (
                - cosB / (W7 * W7) * (
                    W1 * (N1 * cosB - a / rb) * cotB
                    + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                ) / rb * y2[i]
                + cosB / W7 * (
                    (1.0 / rb) * cosB * y2[i] * (N1 * cosB - a / rb) * cotB
                    + W1 * a / pow(rb, 3) * y2[i] * cotB
                    + (2 - 2 * nu) / rb * sinB * y2[i] * cosB
                )
                + 2 * a * y3b * cosB * cotB / pow(rb2, 2) * y2[i]
            )
        ) / PI / (1 - nu))+
        + b2 / 2.0 * (1.0 / 4.0 * (
            N1 * (
                ((2 - 2 * nu) * cotB * cotB + nu) / rb * y2[i] / W6
                - ((2 - 2 * nu) * cotB * cotB + 1) * cosB / rb * y2[i] / W7
            )
            - N1 / (W6 * W6) * (
                -N1 * y1[i] * cotB
                + nu * y3b
                - a
                + a * y1[i] * cotB / rb
                + y1[i] * y1[i] / W6 * W4
            ) / rb * y2[i]
            + N1 / W6 * (
                - a * y1[i] * cotB / (rb * rb * rb) * y2[i]
                - y1[i] * y1[i] / (W6 * W6) * W4 / rb * y2[i]
                - y1[i] * y1[i] / W6 * a / (rb * rb * rb) * y2[i]
            )
            + N1 * cotB / (W7 * W7) * (z1b * cosB - a * (rb * sinB - y1[i]) / rb / cosB) / rb * y2[i]
            - N1 * cotB / W7 * (
                - a / rb2 * sinB * y2[i] / cosB
                + a * (rb * sinB - y1[i]) / (rb * rb * rb) / cosB * y2[i]
            )
            + 3 * a * y2[i] * W8 * cotB / (rb * rb * rb * rb * rb) * y1[i]
            - W8 / (W6 * W6) * (
                2 * nu
                + 1.0 / rb * (N1 * y1[i] * cotB + a)
                - y1[i] * y1[i] / rb / W6 * W5
                - a * y1[i] * y1[i] / (rb * rb * rb)
            ) / rb * y2[i]
            + W8 / W6 * (
                - 1.0 / (rb * rb * rb) * (N1 * y1[i] * cotB + a) * y2[i]
                + y1[i] * y1[i] / (rb * rb * rb) / W6 * W5 * y2[i]
                + y1[i] * y1[i] / rb2 / (W6 * W6) * W5 * y2[i]
                + y1[i] * y1[i] / (rb2 * rb2) / W6 * a * y2[i]
                + 3 * a * y1[i] * y1[i] / (rb * rb * rb * rb * rb) * y2[i]
            )
            - W8 * cotB / (W7 * W7) * (
                - cosB * sinB
                + a * y1[i] * y3b / (rb * rb * rb) / cosB
                + (rb * sinB - y1[i]) / rb * ((2 - 2 * nu) * cosB - W1 / W7 * W9)
            ) / rb * y2[i]
            + W8 * cotB / W7 * (
                - 3 * a * y1[i] * y3b / (rb * rb * rb * rb * rb) / cosB * y2[i]
                + 1.0 / rb2 * sinB * y2[i] * ((2 - 2 * nu) * cosB - W1 / W7 * W9)
                - (rb * sinB - y1[i]) / (rb * rb * rb) * ((2 - 2 * nu) * cosB - W1 / W7 * W9) * y2[i]
                + (rb * sinB - y1[i]) / rb * (
                    - 1.0 / rb * cosB * y2[i] / W7 * W9
                    + W1 / (W7 * W7) * W9 / rb * y2[i]
                    + W1 / W7 * a / (rb * rb * rb) / cosB * y2[i]
                )
            )
        ) / PI / (1 - nu))+
        b3 / 2 * (1.0 / 4.0 * (
            N1 * (
                1.0 / W6 * (1 + a / rb)
                - y2[i] * y2[i] / (W6 * W6) * (1 + a / rb) / rb
                - y2[i] * y2[i] / W6 * a / (rb * rb * rb)
                - cosB / W7 * W2
                + y2[i] * y2[i] * cosB / (W7 * W7) * W2 / rb
                + y2[i] * y2[i] * cosB / W7 * a / (rb * rb * rb)
            )
            - W8 / rb * (a / rb2 + 1.0 / W6)
            + y2[i] * y2[i] * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6)
            - y2[i] * W8 / rb * (
                -2 * a / (rb2 * rb2) * y2[i]
                - 1.0 / (W6 * W6) / rb * y2[i]
            )
            + W8 * cosB / rb / W7 * (W1 / W7 * W2 + a * y3b / rb2)
            - y2[i] * y2[i] * W8 * cosB / (rb * rb * rb) / W7 * (W1 / W7 * W2 + a * y3b / rb2)
            - y2[i] * y2[i] * W8 * cosB / (rb2 * W7 * W7) * (W1 / W7 * W2 + a * y3b / rb2)
            + y2[i] * W8 * cosB / rb / W7 * (
                1.0 / rb * cosB * y2[i] / W7 * W2
                - W1 / (W7 * W7) * W2 / rb * y2[i]
                - W1 / W7 * a / (rb * rb * rb) * y2[i]
                - 2 * a * y3b / (rb2 * rb2) * y2[i]
            )
        ) / PI / (1 - nu))+
        b1 / 2 * (1.0 / 4.0 * (
            N1 * (
                (((2 - 2 * nu) * cotB * cotB - nu) / rb * y1[i] / W6)
                - ((2 - 2 * nu) * cotB * cotB + 1 - 2 * nu) * cosB * (y1[i] / rb - sinB) / W7
            )
            + N1 / (W6 * W6) * (y1[i] * cotB * (1 - W5) + nu * y3b - a + y2[i] * y2[i] / W6 * W4) / rb * y1[i]
            - N1 / W6 * (
                (1 - W5) * cotB
                + a * y1[i] * y1[i] * cotB / (rb * rb * rb)
                - y2[i] * y2[i] / (W6 * W6) * W4 / rb * y1[i]
                - y2[i] * y2[i] / W6 * a / (rb * rb * rb) * y1[i]
            )
            - N1 * cosB * cotB / W7 * W2
            + N1 * z1b * cotB / (W7 * W7) * W2 * (y1[i] / rb - sinB)
            + N1 * z1b * cotB / W7 * a / (rb * rb * rb) * y1[i]
            - a * W8 * cotB / (rb * rb * rb)
            + 3 * a * y1[i] * y1[i] * W8 * cotB / (rb * rb * rb * rb * rb)
            - W8 / (W6 * W6) * (
                -2 * nu + 1.0 / rb * (N1 * y1[i] * cotB - a) + y2[i] * y2[i] / rb / W6 * W5 + a * y2[i] * y2[i] / (rb * rb * rb)
            ) / rb * y1[i]
            + W8 / W6 * (
                -1.0 / (rb * rb * rb) * (N1 * y1[i] * cotB - a) * y1[i]
                + 1.0 / rb * N1 * cotB
                - y2[i] * y2[i] / (rb * rb * rb) / W6 * W5 * y1[i]
                - y2[i] * y2[i] / (rb2 * W6 * W6) * W5 * y1[i]
                - y2[i] * y2[i] / (rb2 * rb2) / W6 * a * y1[i]
                - 3 * a * y2[i] * y2[i] / (rb * rb * rb * rb * rb) * y1[i]
            )
            - W8 / (W7 * W7) * (
                cosB * cosB
                - 1.0 / rb * (N1 * z1b * cotB + a * cosB)
                + a * y3b * z1b * cotB / (rb * rb * rb)
                - 1.0 / rb / W7 * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1)
            ) * (y1[i] / rb - sinB)
            + W8 / W7 * (
                1.0 / (rb * rb * rb) * (N1 * z1b * cotB + a * cosB) * y1[i]
                - 1.0 / rb * N1 * cosB * cotB
                + a * y3b * cosB * cotB / (rb * rb * rb)
                - 3 * a * y3b * z1b * cotB / (rb * rb * rb * rb * rb) * y1[i]
                + 1.0 / (rb * rb * rb) / W7 * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1) * y1[i]
                + 1.0 / rb / (W7 * W7) * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1) * (y1[i] / rb - sinB)
                - 1.0 / rb / W7 * (
                    -a * cosB * cotB / rb * W1
                    + a * z1b * cotB / (rb * rb * rb) * W1 * y1[i]
                    - a * z1b * cotB / rb2 * cosB * y1[i]
                )
            )
        ) / PI / (1 - nu))+
        b2 / 2 * (1.0 / 4.0 * (
            (2 - 2 * nu) * N1 * rFib_ry1 * cotB * cotB
            - N1 * y2[i] / (W6 * W6) * ((W5 - 1) * cotB + y1[i] / W6 * W4) / rb * y1[i]
            + N1 * y2[i] / W6 * (
                -a / (rb * rb * rb) * y1[i] * cotB
                + 1.0 / W6 * W4
                - y1[i] * y1[i] / (W6 * W6) * W4 / rb
                - y1[i] * y1[i] / W6 * a / (rb * rb * rb)
            )
            + N1 * y2[i] * cotB / (W7 * W7) * W9 * (y1[i] / rb - sinB)
            + N1 * y2[i] * cotB / W7 * a / (rb * rb * rb) / cosB * y1[i]
            + 3 * a * y2[i] * W8 * cotB / (rb * rb * rb * rb * rb) * y1[i]
            - y2[i] * W8 / (rb * rb * rb) / W6 * (
                N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb * (1.0 / rb + 1.0 / W6)
            ) * y1[i]
            - y2[i] * W8 / rb2 / (W6 * W6) * (
                N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb * (1.0 / rb + 1.0 / W6)
            ) * y1[i]
            + y2[i] * W8 / rb / W6 * (
                -2 * nu / W6
                + 2 * nu * y1[i] * y1[i] / (W6 * W6) / rb
                - a / rb * (1.0 / rb + 1.0 / W6)
                + a * y1[i] * y1[i] / (rb * rb * rb) * (1.0 / rb + 1.0 / W6)
                - a * y1[i] / rb * (-1.0 / (rb * rb * rb) * y1[i] - 1.0 / (W6 * W6) / rb * y1[i])
            )
            - y2[i] * W8 * cotB / (rb * rb * rb) / W7 * (
                (-2 + 2 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB
            ) * y1[i]
            - y2[i] * W8 * cotB / rb / (W7 * W7) * (
                (-2 + 2 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB
            ) * (y1[i] / rb - sinB)
            + y2[i] * W8 * cotB / rb / W7 * (
                1.0 / rb * cosB * y1[i] / W7 * W9
                - W1 / (W7 * W7) * W9 * (y1[i] / rb - sinB)
                - W1 / W7 * a / (rb * rb * rb) / cosB * y1[i]
                - 2 * a * y3b / (rb2 * rb2) / cosB * y1[i]
            )
        ) / PI / (1 - nu))+
        b3 / 2 * (1.0 / 4.0 * (
            N1 * (
                -sinB * (y1[i] / rb - sinB) / W7
                - 1.0 / W6 * (1.0 + a / rb)
                + y1[i] * y1[i] / (W6 * W6) * (1.0 + a / rb) / rb
                + y1[i] * y1[i] / W6 * a / (rb * rb * rb)
                + cosB / W7 * W2
                - z1b / (W7 * W7) * W2 * (y1[i] / rb - sinB)
                - z1b / W7 * a / (rb * rb * rb) * y1[i]
            )
            + W8 / rb * (a / rb2 + 1.0 / W6)
            - y1[i] * y1[i] * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6)
            + y1[i] * W8 / rb * (
                -2 * a / (rb2 * rb2) * y1[i]
                - 1.0 / (W6 * W6) / rb * y1[i]
            )
            + W8 / (W7 * W7) * (
                sinB * (cosB - a / rb)
                + z1b / rb * (1.0 + a * y3b / rb2)
                - 1.0 / rb / W7 * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1)
            ) * (y1[i] / rb - sinB)
            - W8 / W7 * (
                sinB * a / (rb * rb * rb) * y1[i]
                + cosB / rb * (1.0 + a * y3b / rb2)
                - z1b / (rb * rb * rb) * (1.0 + a * y3b / rb2) * y1[i]
                - 2 * z1b / (rb * rb * rb * rb * rb) * a * y3b * y1[i]
                + 1.0 / (rb * rb * rb) / W7 * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1) * y1[i]
                + 1.0 / rb / (W7 * W7) * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1) * (y1[i] / rb - sinB)
                - 1.0 / rb / W7 * (
                    -a * cosB / rb * W1
                    + a * z1b / (rb * rb * rb) * W1 * y1[i]
                    - a * z1b / rb2 * cosB * y1[i]
                )
            )
        ) / PI / (1.0 - nu));


        v13[i]=b1 / 2 * (1.0 / 4.0 * (
                (-2 + 2 * nu) * N1 * rFib_ry3 * cotB * cotB
                - N1 * y2[i] / (W6 * W6) * ((1 - W5) * cotB - y1[i] / W6 * W4) * (y3b / rb + 1)
                + N1 * y2[i] / W6 * (
                    0.5 * a / (rb * rb * rb) * 2 * y3b * cotB
                    + y1[i] / (W6 * W6) * W4 * (y3b / rb + 1)
                    + 0.5 * y1[i] / W6 * a / (rb * rb * rb) * 2 * y3b
                )
                - N1 * y2[i] * cosB * cotB / (W7 * W7) * W2 * W3
                - 0.5 * N1 * y2[i] * cosB * cotB / W7 * a / (rb * rb * rb) * 2 * y3b
                + a / (rb * rb * rb) * y2[i] * cotB
                - 1.5 * a * y2[i] * W8 * cotB / (rb * rb * rb * rb * rb) * 2 * y3b
                + y2[i] / rb / W6 * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2)
                - 0.5 * y2[i] * W8 / (rb * rb * rb) / W6 * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2) * 2 * y3b
                - y2[i] * W8 / rb / (W6 * W6) * (-N1 * cotB + y1[i] / W6 * W5 + a * y1[i] / rb2) * (y3b / rb + 1)
                + y2[i] * W8 / rb / W6 * (
                    - y1[i] / (W6 * W6) * W5 * (y3b / rb + 1)
                    - 0.5 * y1[i] / W6 * a / (rb * rb * rb) * 2 * y3b
                    - a * y1[i] / (rb2 * rb2) * 2 * y3b
                )
                + y2[i] / rb / W7 * (
                    cosB / W7 * (
                        W1 * (N1 * cosB - a / rb) * cotB
                        + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                    )
                    - a * y3b * cosB * cotB / rb2
                )
                - 0.5 * y2[i] * W8 / (rb * rb * rb) / W7 * (
                    cosB / W7 * (
                        W1 * (N1 * cosB - a / rb) * cotB
                        + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                    )
                    - a * y3b * cosB * cotB / rb2
                ) * 2 * y3b
                - y2[i] * W8 / rb / (W7 * W7) * (
                    cosB / W7 * (
                        W1 * (N1 * cosB - a / rb) * cotB
                        + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                    )
                    - a * y3b * cosB * cotB / rb2
                ) * W3
                + y2[i] * W8 / rb / W7 * (
                    - cosB / (W7 * W7) * (
                        W1 * (N1 * cosB - a / rb) * cotB
                        + (2 - 2 * nu) * (rb * sinB - y1[i]) * cosB
                    ) * W3
                    + cosB / W7 * (
                        (cosB * y3b / rb + 1) * (N1 * cosB - a / rb) * cotB
                        + 0.5 * W1 * a / (rb * rb * rb) * 2 * y3b * cotB
                        + 0.5 * (2 - 2 * nu) / rb * sinB * 2 * y3b * cosB
                    )
                    - a * cosB * cotB / rb2
                    + a * y3b * cosB * cotB / (rb2 * rb2) * 2 * y3b
                )
            ) / PI / (1.0 - nu))+
            b2 / 2 * (1.0 / 4.0 * (
                N1 * (
                    ((2 - 2 * nu) * cotB * cotB + nu) * (y3b / rb + 1) / W6
                    - ((2 - 2 * nu) * cotB * cotB + 1) * cosB * W3 / W7
                )
                - N1 / (W6 * W6) * (
                    -N1 * y1[i] * cotB + nu * y3b - a + a * y1[i] * cotB / rb + y1[i] * y1[i] / W6 * W4
                ) * (y3b / rb + 1)
                + N1 / W6 * (
                    nu
                    - 0.5 * a * y1[i] * cotB / (rb * rb * rb) * 2.0 * y3b
                    - y1[i] * y1[i] / (W6 * W6) * W4 * (y3b / rb + 1)
                    - 0.5 * y1[i] * y1[i] / W6 * a / (rb * rb * rb) * 2.0 * y3b
                )
                + N1 * cotB / (W7 * W7) * (z1b * cosB - a * (rb * sinB - y1[i]) / rb / cosB) * W3
                - N1 * cotB / W7 * (
                    cosB * sinB
                    - 0.5 * a / rb2 * sinB * 2.0 * y3b / cosB
                    + 0.5 * a * (rb * sinB - y1[i]) / (rb * rb * rb) / cosB * 2.0 * y3b
                )
                - a / (rb * rb * rb) * y1[i] * cotB
                + 1.5 * a * y1[i] * W8 * cotB / (rb * rb * rb * rb * rb) * 2.0 * y3b
                + 1.0 / W6 * (
                    2 * nu
                    + 1.0 / rb * (N1 * y1[i] * cotB + a)
                    - y1[i] * y1[i] / rb / W6 * W5
                    - a * y1[i] * y1[i] / (rb * rb * rb)
                )
                - W8 / (W6 * W6) * (
                    2 * nu
                    + 1.0 / rb * (N1 * y1[i] * cotB + a)
                    - y1[i] * y1[i] / rb / W6 * W5
                    - a * y1[i] * y1[i] / (rb * rb * rb)
                ) * (y3b / rb + 1)
                + W8 / W6 * (
                    -0.5 / (rb * rb * rb) * (N1 * y1[i] * cotB + a) * 2.0 * y3b
                    + 0.5 * y1[i] * y1[i] / (rb * rb * rb) / W6 * W5 * 2.0 * y3b
                    + y1[i] * y1[i] / rb / (W6 * W6) * W5 * (y3b / rb + 1)
                    + 0.5 * y1[i] * y1[i] / (rb2 * rb2) / W6 * a * 2.0 * y3b
                    + 1.5 * a * y1[i] * y1[i] / (rb * rb * rb * rb * rb) * 2.0 * y3b
                )
                + cotB / W7 * (
                    -cosB * sinB
                    + a * y1[i] * y3b / (rb * rb * rb) / cosB
                    + (rb * sinB - y1[i]) / rb * ((2 - 2 * nu) * cosB - W1 / W7 * W9)
                )
                - W8 * cotB / (W7 * W7) * (
                    -cosB * sinB
                    + a * y1[i] * y3b / (rb * rb * rb) / cosB
                    + (rb * sinB - y1[i]) / rb * ((2 - 2 * nu) * cosB - W1 / W7 * W9)
                ) * W3
                + W8 * cotB / W7 * (
                    a / (rb * rb * rb) / cosB * y1[i]
                    - 1.5 * a * y1[i] * y3b / (rb * rb * rb * rb * rb) / cosB * 2.0 * y3b
                    + 0.5 / rb2 * sinB * 2.0 * y3b * ((2 - 2 * nu) * cosB - W1 / W7 * W9)
                    - 0.5 * (rb * sinB - y1[i]) / (rb * rb * rb) * ((2 - 2 * nu) * cosB - W1 / W7 * W9) * 2.0 * y3b
                    + (rb * sinB - y1[i]) / rb * (
                        -(cosB * y3b / rb + 1) / W7 * W9
                        + W1 / (W7 * W7) * W9 * W3
                        + 0.5 * W1 / W7 * a / (rb * rb * rb) / cosB * 2.0 * y3b
                    )
                )
            ) / PI / (1.0 - nu))+
        b3 / 2.0 * (1.0 / 4.0 * (
            N1 * (
                - y2[i] / (W6 * W6) * (1.0 + a / rb) * (y3b / rb + 1.0)
                - 0.5 * y2[i] / W6 * a / (rb * rb * rb) * 2.0 * y3b
                + y2[i] * cosB / (W7 * W7) * W2 * W3
                + 0.5 * y2[i] * cosB / W7 * a / (rb * rb * rb) * 2.0 * y3b
            )
            - y2[i] / rb * (a / rb2 + 1.0 / W6)
            + 0.5 * y2[i] * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) * 2.0 * y3b
            - y2[i] * W8 / rb * (- a / (rb2 * rb2) * 2.0 * y3b - 1.0 / (W6 * W6) * (y3b / rb + 1.0))
            + y2[i] * cosB / rb / W7 * (W1 / W7 * W2 + a * y3b / rb2)
            - 0.5 * y2[i] * W8 * cosB / (rb * rb * rb) / W7 * (W1 / W7 * W2 + a * y3b / rb2) * 2.0 * y3b
            - y2[i] * W8 * cosB / rb / (W7 * W7) * (W1 / W7 * W2 + a * y3b / rb2) * W3
            + y2[i] * W8 * cosB / rb / W7 * (
                (cosB * y3b / rb + 1.0) / W7 * W2
                - W1 / (W7 * W7) * W2 * W3
                - 0.5 * W1 / W7 * a / (rb * rb * rb) * 2.0 * y3b
                + a / rb2
                - a * y3b / (rb2 * rb2) * 2.0 * y3b
            )
        ) / PI / (1.0 - nu))+
        b1 / 2.0 * (1.0 / 4.0 * (
            (2.0 - 2.0 * nu) * (
                N1 * rFib_ry1 * cotB
                - y1[i] / (W6 * W6) * W5 / rb * y2[i]
                - y2[i] / W6 * a / (rb * rb * rb) * y1[i]
                + y2[i] * cosB / (W7 * W7) * W2 * (y1[i] / rb - sinB)
                + y2[i] * cosB / W7 * a / (rb * rb * rb) * y1[i]
            )
            - y2[i] * W8 / (rb * rb * rb) * (2.0 * nu / W6 + a / rb2) * y1[i]
            + y2[i] * W8 / rb * (
                -2.0 * nu / (W6 * W6) / rb * y1[i]
                -2.0 * a / (rb2 * rb2) * y1[i]
            )
            - y2[i] * W8 * cosB / (rb * rb * rb) / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) * y1[i]
            - y2[i] * W8 * cosB / rb / (W7 * W7) * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) * (y1[i] / rb - sinB)
            + y2[i] * W8 * cosB / rb / W7 * (
                -1.0 / rb * cosB * y1[i] / W7 * W2
                + W1 / (W7 * W7) * W2 * (y1[i] / rb - sinB)
                + W1 / W7 * a / (rb * rb * rb) * y1[i]
                + 2.0 * a * y3b / (rb2 * rb2) * y1[i]
            )
        ) / PI / (1.0 - nu))+
        b2 / 2.0 * (1.0 / 4.0 * (
            (-2.0 + 2.0 * nu) * N1 * cotB * (
                1.0 / rb * y1[i] / W6 - cosB * (y1[i] / rb - sinB) / W7
            )
            - (2.0 - 2.0 * nu) / W6 * W5
            + (2.0 - 2.0 * nu) * y1[i] * y1[i] / (W6 * W6) * W5 / rb
            + (2.0 - 2.0 * nu) * y1[i] * y1[i] / W6 * a / (rb * rb * rb)
            + (2.0 - 2.0 * nu) * cosB / W7 * W2
            - (2.0 - 2.0 * nu) * z1b / (W7 * W7) * W2 * (y1[i] / rb - sinB)
            - (2.0 - 2.0 * nu) * z1b / W7 * a / (rb * rb * rb) * y1[i]
            - W8 / (rb * rb * rb) * (N1 * cotB - 2.0 * nu * y1[i] / W6 - a * y1[i] / rb2) * y1[i]
            + W8 / rb * (
                -2.0 * nu / W6 + 2.0 * nu * y1[i] * y1[i] / (W6 * W6) / rb
                - a / rb2 + 2.0 * a * y1[i] * y1[i] / (rb2 * rb2)
            )
            + W8 / (W7 * W7) * (
                cosB * sinB
                + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7)
                + a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)
            ) * (y1[i] / rb - sinB)
            - W8 / W7 * (
                1.0 / rb2 * cosB * y1[i] * cotB * ((2.0 - 2.0 * nu) * cosB - W1 / W7)
                - W1 * cotB / (rb * rb * rb) * ((2.0 - 2.0 * nu) * cosB - W1 / W7) * y1[i]
                + W1 * cotB / rb * (
                    -1.0 / rb * cosB * y1[i] / W7 + W1 / (W7 * W7) * (y1[i] / rb - sinB)
                )
                - a / (rb * rb * rb) * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7) * y1[i]
                + a / rb * (
                    -y3b * cosB / rb2
                    + 2.0 * y3b * z1b / (rb2 * rb2) * y1[i]
                    - cosB * W1 / rb / W7
                    - z1b / rb2 * cosB * y1[i] / W7
                    + z1b * W1 / (rb * rb * rb) / W7 * y1[i]
                    + z1b * W1 / rb / (W7 * W7) * (y1[i] / rb - sinB)
                )
            )
        ) / PI / (1.0 - nu))+
            b3 / 2 * (1.0 / 4 * ((2 - 2 * nu) * rFib_ry1
            - (2 - 2 * nu) * y2[i] * sinB / (W7 * W7) * W2 * (y1[i] / rb - sinB)
            - (2 - 2 * nu) * y2[i] * sinB / W7 * a / (rb * rb * rb) * y1[i]
            - y2[i] * W8 * sinB / (rb * rb * rb) / W7 * (1 + W1 / W7 * W2 + a * y3b / rb2) * y1[i]
            - y2[i] * W8 * sinB / rb / (W7 * W7) * (1 + W1 / W7 * W2 + a * y3b / rb2) * (y1[i] / rb - sinB)
            + y2[i] * W8 * sinB / rb / W7 * (1 / rb * cosB * y1[i] / W7 * W2
            - W1 / (W7 * W7) * W2 * (y1[i] / rb - sinB)
            - W1 / W7 * a / (rb * rb * rb) * y1[i]
            - 2 * a * y3b / (rb2 * rb2) * y1[i]))
            / PI / (1 - nu));

        v23[i]=b1 / 2 * (1.0 / 4 * (N1 * (((2 - 2 * nu) * cotB * cotB - nu) * (y3b / rb + 1) / W6
            - ((2 - 2 * nu) * cotB * cotB + 1 - 2 * nu) * cosB * W3 / W7)
            + N1 / (W6 * W6) * (y1[i] * cotB * (1 - W5) + nu * y3b - a + y2[i] * y2[i] / W6 * W4) * (y3b / rb + 1)
            - N1 / W6 * (0.5 * a * y1[i] * cotB / (rb * rb * rb) * 2 * y3b + nu
            - y2[i] * y2[i] / (W6 * W6) * W4 * (y3b / rb + 1)
            - 0.5 * y2[i] * y2[i] / W6 * a / (rb * rb * rb) * 2 * y3b)
            - N1 * sinB * cotB / W7 * W2
            + N1 * z1b * cotB / (W7 * W7) * W2 * W3
            + 0.5 * N1 * z1b * cotB / W7 * a / (rb * rb * rb) * 2 * y3b
            - a / (rb * rb * rb) * y1[i] * cotB
            + 1.5 * a * y1[i] * W8 * cotB / (rb * rb * rb * rb * rb) * 2 * y3b
            + 1 / W6 * (-2 * nu + 1 / rb * (N1 * y1[i] * cotB - a)
            + y2[i] * y2[i] / rb / W6 * W5 + a * y2[i] * y2[i] / (rb * rb * rb))
            - W8 / (W6 * W6) * (-2 * nu + 1 / rb * (N1 * y1[i] * cotB - a)
            + y2[i] * y2[i] / rb / W6 * W5 + a * y2[i] * y2[i] / (rb * rb * rb)) * (y3b / rb + 1)
            + W8 / W6 * (-0.5 / (rb * rb * rb) * (N1 * y1[i] * cotB - a) * 2 * y3b
            - 0.5 * y2[i] * y2[i] / (rb * rb * rb) / W6 * W5 * 2 * y3b
            - y2[i] * y2[i] / rb / (W6 * W6) * W5 * (y3b / rb + 1)
            - 0.5 * y2[i] * y2[i] / (rb2 * rb2) / W6 * a * 2 * y3b
            - 1.5 * a * y2[i] * y2[i] / (rb * rb * rb * rb * rb) * 2 * y3b)
            + 1 / W7 * (cosB * cosB - 1 / rb * (N1 * z1b * cotB + a * cosB)
            + a * y3b * z1b * cotB / (rb * rb * rb)
            - 1 / rb / W7 * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1))
            - W8 / (W7 * W7) * (cosB * cosB - 1 / rb * (N1 * z1b * cotB + a * cosB)
            + a * y3b * z1b * cotB / (rb * rb * rb)
            - 1 / rb / W7 * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1)) * W3
            + W8 / W7 * (0.5 / (rb * rb * rb) * (N1 * z1b * cotB + a * cosB) * 2 * y3b
            - 1 / rb * N1 * sinB * cotB
            + a * z1b * cotB / (rb * rb * rb)
            + a * y3b * sinB * cotB / (rb * rb * rb)
            - 1.5 * a * y3b * z1b * cotB / (rb * rb * rb * rb * rb) * 2 * y3b
            + 0.5 / (rb * rb * rb) / W7 * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1) * 2 * y3b
            + 1 / rb / (W7 * W7) * (y2[i] * y2[i] * cosB * cosB - a * z1b * cotB / rb * W1) * W3
            - 1 / rb / W7 * (-a * sinB * cotB / rb * W1
            + 0.5 * a * z1b * cotB / (rb * rb * rb) * W1 * 2 * y3b
            - a * z1b * cotB / rb * (cosB * y3b / rb + 1))))
            / PI / (1 - nu))+
        b2 / 2 * (1.0 / 4 * ((2 - 2 * nu) * N1 * rFib_ry3 * cotB * cotB
            - N1 * y2[i] / (W6 * W6) * ((W5 - 1) * cotB + y1[i] / W6 * W4) * (y3b / rb + 1)
            + N1 * y2[i] / W6 * (-0.5 * a / pow(rb, 3) * 2 * y3b * cotB
            - y1[i] / (W6 * W6) * W4 * (y3b / rb + 1)
            - 0.5 * y1[i] / W6 * a / pow(rb, 3) * 2 * y3b)
            + N1 * y2[i] * cotB / (W7 * W7) * W9 * W3
            + 0.5 * N1 * y2[i] * cotB / W7 * a / pow(rb, 3) / cosB * 2 * y3b
            - a / pow(rb, 3) * y2[i] * cotB
            + 1.5 * a * y2[i] * W8 * cotB / pow(rb, 5) * 2 * y3b
            + y2[i] / rb / W6 * (N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb * (1 / rb + 1 / W6))
            - 0.5 * y2[i] * W8 / pow(rb, 3) / W6 * (N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb * (1 / rb + 1 / W6)) * 2 * y3b
            - y2[i] * W8 / rb / (W6 * W6) * (N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb * (1 / rb + 1 / W6)) * (y3b / rb + 1)
            + y2[i] * W8 / rb / W6 * (2 * nu * y1[i] / (W6 * W6) * (y3b / rb + 1)
            + 0.5 * a * y1[i] / pow(rb, 3) * (1 / rb + 1 / W6) * 2 * y3b
            - a * y1[i] / rb * (-0.5 / pow(rb, 3) * 2 * y3b - 1 / (W6 * W6) * (y3b / rb + 1)))
            + y2[i] * cotB / rb / W7 * ((-2 + 2 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB)
            - 0.5 * y2[i] * W8 * cotB / pow(rb, 3) / W7 * ((-2 + 2 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) * 2 * y3b
            - y2[i] * W8 * cotB / rb / (W7 * W7) * ((-2 + 2 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) * W3
            + y2[i] * W8 * cotB / rb / W7 * ((cosB * y3b / rb + 1) / W7 * W9
            - W1 / (W7 * W7) * W9 * W3
            - 0.5 * W1 / W7 * a / pow(rb, 3) / cosB * 2 * y3b
            + a / rb2 / cosB
            - a * y3b / (rb2 * rb2) / cosB * 2 * y3b))) / PI / (1 - nu)+
        b3 / 2 * (1.0 / 4 * (N1 * (
            -sinB * W3 / W7
            + y1[i] / (W6 * W6) * (1 + a / rb) * (y3b / rb + 1)
            + 0.5 * y1[i] / W6 * a / pow(rb, 3) * 2 * y3b
            + sinB / W7 * W2
            - z1b / (W7 * W7) * W2 * W3
            - 0.5 * z1b / W7 * a / pow(rb, 3) * 2 * y3b)
            + y1[i] / rb * (a / rb2 + 1 / W6)
            - 0.5 * y1[i] * W8 / pow(rb, 3) * (a / rb2 + 1 / W6) * 2 * y3b
            + y1[i] * W8 / rb * (-a / (rb2 * rb2) * 2 * y3b - 1 / (W6 * W6) * (y3b / rb + 1))
            - 1 / W7 * (sinB * (cosB - a / rb) + z1b / rb * (1 + a * y3b / rb2) - 1.0 / rb / W7 * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1))
            + W8 / (W7 * W7) * (sinB * (cosB - a / rb) + z1b / rb * (1 + a * y3b / rb2) - 1 / rb / W7 * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1)) * W3
            - W8 / W7 * (
                0.5 * sinB * a / pow(rb, 3) * 2 * y3b
                + sinB / rb * (1 + a * y3b / rb2)
                - 0.5 * z1b / pow(rb, 3) * (1 + a * y3b / rb2) * 2 * y3b
                + z1b / rb * (a / rb2 - a * y3b / (rb2 * rb2) * 2 * y3b)
                + 0.5 / pow(rb, 3) / W7 * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1) * 2 * y3b
                + 1 / rb / (W7 * W7) * (y2[i] * y2[i] * cosB * sinB - a * z1b / rb * W1) * W3
                - 1.0 / rb / W7 * (-a * sinB / rb * W1 + 0.5 * a * z1b / pow(rb, 3) * W1 * 2 * y3b - a * z1b / rb * (cosB * y3b / rb + 1))))
            / PI / (1 - nu))+
        b1 / 2 * (1.0 / 4 * (
            (2 - 2 * nu) * (
                N1 * rFib_ry2 * cotB
                + 1.0 / W6 * W5
                - y2[i] * y2[i] / (W6 * W6) * W5 / rb
                - y2[i] * y2[i] / W6 * a / pow(rb, 3)
                - cosB / W7 * W2
                + y2[i] * y2[i] * cosB / (W7 * W7) * W2 / rb
                + y2[i] * y2[i] * cosB / W7 * a / pow(rb, 3)
            )
            + W8 / rb * (2 * nu / W6 + a / rb2)
            - y2[i] * y2[i] * W8 / pow(rb, 3) * (2 * nu / W6 + a / rb2)
            + y2[i] * W8 / rb * (-2 * nu / (W6 * W6) / rb * y2[i] - 2 * a / (rb2 * rb2) * y2[i])
            + W8 * cosB / rb / W7 * (1 - 2 * nu - W1 / W7 * W2 - a * y3b / rb2)
            - y2[i] * y2[i] * W8 * cosB / pow(rb, 3) / W7 * (1 - 2 * nu - W1 / W7 * W2 - a * y3b / rb2)
            - y2[i] * y2[i] * W8 * cosB / (rb2 * (W7 * W7)) * (1 - 2 * nu - W1 / W7 * W2 - a * y3b / rb2)
            + y2[i] * W8 * cosB / rb / W7 * (-1.0 / rb * cosB * y2[i] / W7 * W2 + W1 / (W7 * W7) * W2 / rb * y2[i] + W1 / W7 * a / pow(rb, 3) * y2[i] + 2 * a * y3b / (rb2 * rb2) * y2[i])
        ) / PI / (1 - nu))+
        b2 / 2 * (1.0 / 4 * (
            (-2 + 2 * nu) * N1 * cotB * (1.0 / rb * y2[i] / W6 - cosB / rb * y2[i] / W7)
            + (2 - 2 * nu) * y1[i] / (W6 * W6) * W5 / rb * y2[i]
            + (2 - 2 * nu) * y1[i] / W6 * a / pow(rb, 3) * y2[i]
            - (2 - 2 * nu) * z1b / (W7 * W7) * W2 / rb * y2[i]
            - (2 - 2 * nu) * z1b / W7 * a / pow(rb, 3) * y2[i]
            - W8 / pow(rb, 3) * (N1 * cotB - 2 * nu * y1[i] / W6 - a * y1[i] / rb2) * y2[i]
            + W8 / rb * (2 * nu * y1[i] / (W6 * W6) / rb * y2[i] + 2 * a * y1[i] / (rb2 * rb2) * y2[i])
            + W8 / (W7 * W7) * (cosB * sinB + W1 * cotB / rb * ((2 - 2 * nu) * cosB - W1 / W7) + a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)) / rb * y2[i]
            - W8 / W7 * (
                1.0 / (rb2) * cosB * y2[i] * cotB * ((2 - 2 * nu) * cosB - W1 / W7)
                - W1 * cotB / pow(rb, 3) * ((2 - 2 * nu) * cosB - W1 / W7) * y2[i]
                + W1 * cotB / rb * (-cosB / rb * y2[i] / W7 + W1 / (W7 * W7) / rb * y2[i])
                - a / pow(rb, 3) * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7) * y2[i]
                + a / rb * (2 * y3b * z1b / (rb2 * rb2) * y2[i] - z1b / rb2 * cosB * y2[i] / W7 + z1b * W1 / pow(rb, 3) / W7 * y2[i] + z1b * W1 / (rb2 * (W7 * W7)) * y2[i])
            )
        ) / PI / (1 - nu))+
        b3 / 2 * (1.0 / 4 * (
            (2 - 2 * nu) * rFib_ry2
            + (2 - 2 * nu) * sinB / W7 * W2
            - (2 - 2 * nu) * y2[i] * y2[i] * sinB / (W7 * W7) * W2 / rb
            - (2 - 2 * nu) * y2[i] * y2[i] * sinB / W7 * a / pow(rb, 3)
            + W8 * sinB / rb / W7 * (1 + W1 / W7 * W2 + a * y3b / rb2)
            - y2[i] * y2[i] * W8 * sinB / pow(rb, 3) / W7 * (1 + W1 / W7 * W2 + a * y3b / rb2)
            - y2[i] * y2[i] * W8 * sinB / (rb2 * W7 * W7) * (1 + W1 / W7 * W2 + a * y3b / rb2)
            + y2[i] * W8 * sinB / rb / W7 * (
                1.0 / rb * cosB * y2[i] / W7 * W2
                - W1 / (W7 * W7) * W2 / rb * y2[i]
                - W1 / W7 * a / pow(rb, 3) * y2[i]
                - 2 * a * y3b / (rb2 * rb2) * y2[i]
            )
        ) / PI / (1 - nu));






        //;
        //cout<<"v23[i]  "<<v23[i] <<endl;

        //v11[i] = b1 * v11[i] / (PI * (1.0 - nu));
        
    }
}


void AngSetupFSC_S(
    const vector<double>& X,
    const vector<double>& Y,
    const vector<double>& Z,
    const vector<double>& B_vec,
    const vector<double>& PA,
    const vector<double>& PB,
    double mu,
    double lambda_,
    std::vector<double>& Sxx, std::vector<double>& Syy, std::vector<double>& Szz,
    std::vector<double>& Sxy, std::vector<double>& Sxz, std::vector<double>& Syz,
    std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
    std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz) 
{

    // Compute Poisson's ratio
    double nu = 1.0 / (1.0 + lambda_ / mu) / 2.0;

    // Compute vector from PA to PB
    Vec3 SideVec = {
        PB[0] - PA[0],
        PB[1] - PA[1],
        PB[2] - PA[2]
    };

    // Define unit vector along Z-axis
    Vec3 eZ = {0.0, 0.0, 1.0};

    // Compute angle beta between -SideVec and Z-axis
    double dotVal = -(SideVec[0]*eZ[0] + SideVec[1]*eZ[1] + SideVec[2]*eZ[2]);
    double normSide = sqrt(SideVec[0]*SideVec[0] + SideVec[1]*SideVec[1] + SideVec[2]*SideVec[2]);
    double beta = acos(dotVal / normSide);

    //double eps = 2.2204e-16;
    double eps = 1.0e-4;
    size_t N = X.size();
    if (fabs(beta) < eps || fabs(PI - beta) < eps) {
        // Special case: dislocation is vertical
        size_t n = X.size();
        // vector<vector<double>> Stress(6, vector<double>(n, 0.0));
        // vector<vector<double>> Strain(6, vector<double>(n, 0.0));
        Sxx.resize(N);
        Syy.resize(N);
        Szz.resize(N);
        Sxy.resize(N);
        Sxz.resize(N);
        Syz.resize(N);
        Exx.resize(N);
        Eyy.resize(N);
        Ezz.resize(N);
        Exy.resize(N);
        Exz.resize(N);
        Eyz.resize(N);
        for (size_t i = 0; i < N; ++i)
        {
            Sxx[i] = 0;Syy[i] = 0;Szz[i] = 0;
            Sxy[i] = 0;Sxz[i] = 0;Syz[i] = 0;
            Exx[i] = 0;Eyy[i] = 0;Ezz[i] = 0;
            Exy[i] = 0;Exz[i] = 0;Eyz[i] = 0;
            
        }  
        
        //cout << "Beta is near 0 or pi, Stress and Strain set to zero." << endl;
    } 
    else 
    {
        // Create local coordinate system for angular dislocation

        // ey1 is SideVec projected in XY plane and normalized
        Vec3 ey1 = { SideVec[0], SideVec[1], 0.0 };
        double ey1_len = sqrt(ey1[0]*ey1[0] + ey1[1]*ey1[1]);
        if (ey1_len != 0.0) {
            ey1[0] /= ey1_len;
            ey1[1] /= ey1_len;
        }

        // ey3 is the negative Z-axis
        Vec3 ey3 = { -eZ[0], -eZ[1], -eZ[2] };

        // ey2 = ey3 × ey1 (cross product)
        Vec3 ey2 = {
            ey3[1]*ey1[2] - ey3[2]*ey1[1],
            ey3[2]*ey1[0] - ey3[0]*ey1[2],
            ey3[0]*ey1[1] - ey3[1]*ey1[0]
        };

        // Construct transformation matrix A (columns: ey1, ey2, ey3)
        double A[3][3] = {
            { ey1[0], ey2[0], ey3[0] },
            { ey1[1], ey2[1], ey3[1] },
            { ey1[2], ey2[2], ey3[2] }
        };

        // Transform coordinates from global to local system: yA = (P - PA) * A
        //size_t N = X.size();
        vector<array<double, 3>> yA(N);
        for (size_t i = 0; i < N; ++i) {
            double dx = X[i] - PA[0];
            double dy = Y[i] - PA[1];
            double dz = Z[i] - PA[2];

            yA[i][0] = dx * A[0][0] + dy * A[1][0] + dz * A[2][0];
            yA[i][1] = dx * A[0][1] + dy * A[1][1] + dz * A[2][1];
            yA[i][2] = dx * A[0][2] + dy * A[1][2] + dz * A[2][2];
        }

        // Compute SideVec in the local coordinate system: yAB = SideVec * A
        double yAB[3] = {
            SideVec[0]*A[0][0] + SideVec[1]*A[1][0] + SideVec[2]*A[2][0],
            SideVec[0]*A[0][1] + SideVec[1]*A[1][1] + SideVec[2]*A[2][1],
            SideVec[0]*A[0][2] + SideVec[1]*A[1][2] + SideVec[2]*A[2][2]
        };

        // yB = yA - yAB
        vector<array<double, 3>> yB(N);
        for (size_t i = 0; i < N; ++i) {
            yB[i][0] = yA[i][0] - yAB[0];
            yB[i][1] = yA[i][1] - yAB[1];
            yB[i][2] = yA[i][2] - yAB[2];
        }

        // Transform Burgers vector to local coordinate system: bv = A^T * B_vec
        double bv[3] = {
            A[0][0]*B_vec[0] + A[1][0]*B_vec[1] + A[2][0]*B_vec[2],
            A[0][1]*B_vec[0] + A[1][1]*B_vec[1] + A[2][1]*B_vec[2],
            A[0][2]*B_vec[0] + A[1][2]*B_vec[1] + A[2][2]*B_vec[2]
        };

        // Output bv for verification
        //cout << "bv = (" << bv[0] << ", " << bv[1] << ", " << bv[2] << ")" << endl;
        // Determine the configuration: I = beta * yA_x >= 0
        vector<bool> I(N, false);
        vector<bool> NI(N, false);
        bool judI=false;
        bool judNI=false;
        for (size_t i = 0; i < N; ++i) {
            if (beta * yA[i][0] >= 0) {
                I[i] = true;
                judI=true;
            } else {
                NI[i] = true;
                judNI=true;
            }
            //cout<<I[i]<<" "<<NI[i]<<endl;
        }

        // Initialize strain components for both configurations A and B
        vector<double> v11A(N, 0.0), v22A(N, 0.0), v33A(N, 0.0);
        vector<double> v12A(N, 0.0), v13A(N, 0.0), v23A(N, 0.0);

        vector<double> v11B(N, 0.0), v22B(N, 0.0), v33B(N, 0.0);
        vector<double> v12B(N, 0.0), v13B(N, 0.0), v23B(N, 0.0);

        if(judI==true)
        {
            // Subset inputs where I is true
            vector<double> y1A, y2A, y3A,y1B, y2B, y3B;
            vector<int> indexI;
            for (size_t i = 0; i < I.size(); ++i) {
                if (I[i]) {
                    //cout<<NI[i]<<"  !!!!"<<endl;
                    y1A.push_back(-yA[i][0]); // -yA[:,0]
                    y2A.push_back(-yA[i][1]); // -yA[:,1]
                    y3A.push_back( yA[i][2]); //  yA[:,2]
                    y1B.push_back(-yB[i][0]); // -yB[:,0]
                    y2B.push_back(-yB[i][1]); // -yB[:,1]
                    y3B.push_back( yB[i][2]); //  yB[:,2]
                    indexI.push_back(i);
                }
            }

            // Outputs for strain partials
            vector<double> v11_partA, v22_partA, v33_partA;
            vector<double> v12_partA, v13_partA, v23_partA;
            vector<double> v11_partB, v22_partB, v33_partB;
            vector<double> v12_partB, v13_partB, v23_partB;
            //cout<<"!!!!!"<<endl;
            //AngDisStrainFSC(y1_sub, y2_sub, y3_sub,PI - beta,-bv[0], -bv[1],  bv[2], nu,-PA[2]);
            AngDisStrainFSC(y1A, y2A, y3A,PI-beta,-bv[0], -bv[1],  bv[2], nu,-PA[2],
                            v11_partA, v22_partA, v33_partA,v12_partA, v13_partA, v23_partA);
            
            AngDisStrainFSC(y1B, y2B, y3B,PI-beta,-bv[0], -bv[1],  bv[2], nu,-PB[2],
                            v11_partB, v22_partB, v33_partB,v12_partB, v13_partB, v23_partB);
            
            for (size_t i = 0; i < v11_partA.size(); ++i) {
                //printf("%.20f",v11_partA[i]);
                v13_partA[i]=-v13_partA[i];
                v23_partA[i]=-v23_partA[i];
                v13_partB[i]=-v13_partB[i];
                v23_partB[i]=-v23_partB[i];
                v11A[indexI[i]]=v11_partA[i];v22A[indexI[i]]=v22_partA[i];v33A[indexI[i]]=v33_partA[i];
                v12A[indexI[i]]=v12_partA[i];v13A[indexI[i]]=v13_partA[i];v23A[indexI[i]]=v23_partA[i];
                v11B[indexI[i]]=v11_partB[i];v22B[indexI[i]]=v22_partB[i];v33B[indexI[i]]=v33_partB[i];
                v12B[indexI[i]]=v12_partB[i];v13B[indexI[i]]=v13_partB[i];v23B[indexI[i]]=v23_partB[i];
            }
            
            
        }
        if(judNI==true)
        {
            vector<int> indexNI;
            vector<double> y1A, y2A, y3A,y1B, y2B, y3B;
            for (size_t i = 0; i < NI.size(); ++i) {
                if (NI[i]) {
                    //cout<<i<<" !!!!"<<endl;
                    y1A.push_back( yA[i][0]); // -yA[:,0]
                    y2A.push_back( yA[i][1]); // -yA[:,1]
                    y3A.push_back( yA[i][2]); //  yA[:,2]
                    y1B.push_back( yB[i][0]); // -yB[:,0]
                    y2B.push_back( yB[i][1]); // -yB[:,1]
                    y3B.push_back( yB[i][2]); //  yB[:,2]
                    indexNI.push_back(i);
                }
            }

            // Outputs for strain partials
            vector<double> v11_partA, v22_partA, v33_partA;
            vector<double> v12_partA, v13_partA, v23_partA;
            vector<double> v11_partB, v22_partB, v33_partB;
            vector<double> v12_partB, v13_partB, v23_partB;

            // Call strain calculator (same as AngDisStrainFSC)
            AngDisStrainFSC(y1A, y2A, y3A,beta,bv[0], bv[1],  bv[2], nu,-PA[2],
                        v11_partA, v22_partA, v33_partA,v12_partA, v13_partA, v23_partA);
            
            AngDisStrainFSC(y1B, y2B, y3B,beta,bv[0], bv[1],  bv[2], nu,-PB[2],
                        v11_partB, v22_partB, v33_partB,v12_partB, v13_partB, v23_partB);
            
            for (size_t i = 0; i < v11_partA.size(); ++i) {
                v11A[indexNI[i]]=v11_partA[i];v22A[indexNI[i]]=v22_partA[i];v33A[indexNI[i]]=v33_partA[i];
                v12A[indexNI[i]]=v12_partA[i];v13A[indexNI[i]]=v13_partA[i];v23A[indexNI[i]]=v23_partA[i];
                v11B[indexNI[i]]=v11_partB[i];v22B[indexNI[i]]=v22_partB[i];v33B[indexNI[i]]=v33_partB[i];
                v12B[indexNI[i]]=v12_partB[i];v13B[indexNI[i]]=v13_partB[i];v23B[indexNI[i]]=v23_partB[i];
            }
        }
        Sxx.resize(N);
        Syy.resize(N);
        Szz.resize(N);
        Sxy.resize(N);
        Sxz.resize(N);
        Syz.resize(N);
        Exx.resize(N);
        Eyy.resize(N);
        Ezz.resize(N);
        Exy.resize(N);
        Exz.resize(N);
        Eyz.resize(N);
        for (size_t i = 0; i < I.size(); ++i) 
        {
            if (I[i]) 
            {
                //cout<<"!!!!"<<endl;
                Exx[i]=v11B[i]-v11A[i];
                Eyy[i]=v22B[i]-v22A[i];
                Ezz[i]=v33B[i]-v33A[i];
                Exy[i]=v12B[i]-v12A[i];
                Exz[i]=v13B[i]-v13A[i];
                Eyz[i]=v23B[i]-v23A[i];
            }
            else if(NI[i])
            {
                Exx[i]=v11B[i]-v11A[i];
                Eyy[i]=v22B[i]-v22A[i];
                Ezz[i]=v33B[i]-v33A[i];
                Exy[i]=v12B[i]-v12A[i];
                Exz[i]=v13B[i]-v13A[i];
                Eyz[i]=v23B[i]-v23A[i];
            }

            

	        

            //cout<<v11B[i]<<"  "<<v11A[i]<<endl;
            //printf("%.20f %.20f %.20f %.20f %.20f %.20f\n",v11A[i],v22A[i],v33A[i],v12A[i],v13A[i],v23A[i]);

        }
        TensTrans(Exx,Eyy,Ezz,Exy,Exz,Eyz, A, Exx.size());
        for (size_t i = 0; i < N; ++i)
        {
            double trace = Exx[i] + Eyy[i] + Ezz[i];
            Sxx[i] = 2 * mu * Exx[i] + lambda_ * trace;
            Syy[i] = 2 * mu * Eyy[i] + lambda_ * trace;
            Szz[i] = 2 * mu * Ezz[i] + lambda_ * trace;
            Sxy[i] = 2 * mu * Exy[i];
            Sxz[i] = 2 * mu * Exz[i];
            Syz[i] = 2 * mu * Eyz[i];
        }  
        //AngDisStrainFSC(y1,y2,y3,beta,b1, b2, b3,nu,a)
        //AngDisStrainFSC(-yA[I,0],-yA[I,1],yA[I,2],pi-beta,-bv[0],-bv[1],bv[2],nu,-PA[2])
                                                
                                                
    }
}


void TDstress_HarFunc(const std::vector<double>& X,
                const std::vector<double>& Y,
                const std::vector<double>& Z,
                const std::vector<double>& P1,
                const std::vector<double>& P2,
                const std::vector<double>& P3,
                double Ss, double Ds, double Ts,
                double mu, double lambda_,
                std::vector<double>& StsFSCxx, std::vector<double>& StsFSCyy, std::vector<double>& StsFSCzz,
                std::vector<double>& StsFSCxy, std::vector<double>& StsFSCxz, std::vector<double>& StsFSCyz,
                std::vector<double>& StrFSCxx, std::vector<double>& StrFSCyy, std::vector<double>& StrFSCzz,
                std::vector<double>& StrFSCxy, std::vector<double>& StrFSCxz, std::vector<double>& StrFSCyz) 
{
    // Displacement vector components
    double bx = Ts;
    double by = Ss;
    double bz = Ds;

    // Normal vector
    // Calculate the normal vector Vnorm = normalize(cross(P2 - P1, P3 - P1))
    Vec3 u = {P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]};
    Vec3 v = {P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]};
    Vec3 Vnorm = {
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    };
    double Vnorm_len = sqrt(Vnorm[0]*Vnorm[0] + Vnorm[1]*Vnorm[1] + Vnorm[2]*Vnorm[2]);
    if (Vnorm_len != 0.0) {
        Vnorm[0] /= Vnorm_len;
        Vnorm[1] /= Vnorm_len;
        Vnorm[2] /= Vnorm_len;
    }

    // ----------------------------------------
    // Vstrike = normalize(cross(eZ, Vnorm))
    Vec3 eX = {1.0, 0.0, 0.0};
    Vec3 eY = {0.0, 1.0, 0.0};
    Vec3 eZ = {0.0, 0.0, 1.0};
    Vec3 Vstrike = {
        eZ[1]*Vnorm[2] - eZ[2]*Vnorm[1],
        eZ[2]*Vnorm[0] - eZ[0]*Vnorm[2],
        eZ[0]*Vnorm[1] - eZ[1]*Vnorm[0]
    };
    double Vstrike_len = sqrt(Vstrike[0]*Vstrike[0] + Vstrike[1]*Vstrike[1] + Vstrike[2]*Vstrike[2]);

    if (Vstrike_len <1e-5){
        Vstrike[0] = eX[0] * Vnorm[2];
        Vstrike[1] = eX[1] * Vnorm[2];
        Vstrike[2] = eX[2] * Vnorm[2];
    } else {
        Vstrike[0] /= Vstrike_len;
        Vstrike[1] /= Vstrike_len;
        Vstrike[2] /= Vstrike_len;
    }

    // ----------------------------------------
    // Vdip = cross(Vnorm, Vstrike)
    Vec3 Vdip = {
        Vnorm[1]*Vstrike[2] - Vnorm[2]*Vstrike[1],
        Vnorm[2]*Vstrike[0] - Vnorm[0]*Vstrike[2],
        Vnorm[0]*Vstrike[1] - Vnorm[1]*Vstrike[0]
    };

    // ----------------------------------------
    // rotate A，Vnorm, Vstrike, Vdip
    // B_vec = A * burgers
    std::vector<double> B_vec(3);
    B_vec[0] = Vnorm[0]*bx + Vstrike[0]*by + Vdip[0]*bz;
    B_vec[1] = Vnorm[1]*bx + Vstrike[1]*by + Vdip[1]*bz;
    B_vec[2] = Vnorm[2]*bx + Vstrike[2]*by + Vdip[2]*bz;
    //cout<<B_vec[0]<<"\t"<<B_vec[1]<<"\t"<<B_vec[2]<<endl;

    vector<double> Sxx, Syy, Szz, Sxy, Sxz, Syz;
    vector<double> Exx, Eyy, Ezz, Exy, Exz, Eyz;
    int N=X.size();
    StsFSCxx.resize(N);StsFSCyy.resize(N);StsFSCzz.resize(N);
    StsFSCxy.resize(N);StsFSCxz.resize(N);StsFSCyz.resize(N);
    StrFSCxx.resize(N);StrFSCyy.resize(N);StrFSCzz.resize(N);
    StrFSCxy.resize(N);StrFSCxz.resize(N);StrFSCyz.resize(N);

    AngSetupFSC_S(X,Y,Z,B_vec,P1,P2,mu,lambda_,Sxx, Syy, Szz, Sxy, Sxz, Syz,
                                Exx, Eyy, Ezz, Exy, Exz, Eyz);
    for (size_t i = 0; i < Exx.size(); ++i)
    {
        //cout<<Sxx[i]<<"\t"<<Syy[i]<<"\t"<<Szz[i]<<"\t"<<Sxy[i]<<"\t"<<Sxz[i]<<"\t"<<Syz[i]<<endl;
        StrFSCxx[i]=StrFSCxx[i]+Exx[i];
        StrFSCyy[i]=StrFSCyy[i]+Eyy[i];
        StrFSCzz[i]=StrFSCzz[i]+Ezz[i];
        StrFSCxy[i]=StrFSCxy[i]+Exy[i];
        StrFSCxz[i]=StrFSCxz[i]+Exz[i];
        StrFSCyz[i]=StrFSCyz[i]+Eyz[i];
        StsFSCxx[i]=StsFSCxx[i]+Sxx[i];
        StsFSCyy[i]=StsFSCyy[i]+Syy[i];
        StsFSCzz[i]=StsFSCzz[i]+Szz[i];
        StsFSCxy[i]=StsFSCxy[i]+Sxy[i];
        StsFSCxz[i]=StsFSCxz[i]+Sxz[i];
        StsFSCyz[i]=StsFSCyz[i]+Syz[i];
        // cout<<StsFSCxx[i]<<"\t"<<StsFSCyy[i]<<"\t"<<StsFSCzz[i]<<"\t"
        //     <<StsFSCxy[i]<<"\t"<<StsFSCxz[i]<<"\t"<<StsFSCyz[i]<<endl;
    }
    // for(int i=0;i<3;i++)
    // {
    //     cout<<P2[i]<<"\t"<<P3[i]<<endl;
    // }
    //cout<<B_vec[0]<<"\t"<<B_vec[1]<<"\t"<<B_vec[2]<<endl;
    AngSetupFSC_S(X,Y,Z,B_vec,P2,P3,mu,lambda_,Sxx, Syy, Szz, Sxy, Sxz, Syz,
                                Exx, Eyy, Ezz, Exy, Exz, Eyz);
    for (size_t i = 0; i < Exx.size(); ++i)
    {
        
        StrFSCxx[i]=StrFSCxx[i]+Exx[i];
        StrFSCyy[i]=StrFSCyy[i]+Eyy[i];
        StrFSCzz[i]=StrFSCzz[i]+Ezz[i];
        StrFSCxy[i]=StrFSCxy[i]+Exy[i];
        StrFSCxz[i]=StrFSCxz[i]+Exz[i];
        StrFSCyz[i]=StrFSCyz[i]+Eyz[i];
        StsFSCxx[i]=StsFSCxx[i]+Sxx[i];
        StsFSCyy[i]=StsFSCyy[i]+Syy[i];
        StsFSCzz[i]=StsFSCzz[i]+Szz[i];
        StsFSCxy[i]=StsFSCxy[i]+Sxy[i];
        StsFSCxz[i]=StsFSCxz[i]+Sxz[i];
        StsFSCyz[i]=StsFSCyz[i]+Syz[i];
        // cout<<StsFSCxx[i]<<"\t"<<StsFSCyy[i]<<"\t"<<StsFSCzz[i]<<"\t"
        //     <<StsFSCxy[i]<<"\t"<<StsFSCxz[i]<<"\t"<<StsFSCyz[i]<<endl;
        //cout<<"Stress2   "<<Sxx[i]<<"\t"<<Syy[i]<<"\t"<<Szz[i]<<"\t"<<Sxy[i]<<"\t"<<Sxz[i]<<"\t"<<Syz[i]<<endl;
    }
    AngSetupFSC_S(X,Y,Z,B_vec,P3,P1,mu,lambda_,Sxx, Syy, Szz, Sxy, Sxz, Syz,
                                Exx, Eyy, Ezz, Exy, Exz, Eyz);
    for (size_t i = 0; i < Exx.size(); ++i)
    {
        //cout<<Sxx[i]<<"\t"<<Syy[i]<<"\t"<<Szz[i]<<"\t"<<Sxy[i]<<"\t"<<Sxz[i]<<"\t"<<Syz[i]<<endl;
        StrFSCxx[i]=StrFSCxx[i]+Exx[i];
        StrFSCyy[i]=StrFSCyy[i]+Eyy[i];
        StrFSCzz[i]=StrFSCzz[i]+Ezz[i];
        StrFSCxy[i]=StrFSCxy[i]+Exy[i];
        StrFSCxz[i]=StrFSCxz[i]+Exz[i];
        StrFSCyz[i]=StrFSCyz[i]+Eyz[i];
        StsFSCxx[i]=StsFSCxx[i]+Sxx[i];
        StsFSCyy[i]=StsFSCyy[i]+Syy[i];
        StsFSCzz[i]=StsFSCzz[i]+Szz[i];
        StsFSCxy[i]=StsFSCxy[i]+Sxy[i];
        StsFSCxz[i]=StsFSCxz[i]+Sxz[i];
        StsFSCyz[i]=StsFSCyz[i]+Syz[i];
        // cout<<StsFSCxx[i]<<"\t"<<StsFSCyy[i]<<"\t"<<StsFSCzz[i]<<"\t"
        //     <<StsFSCxy[i]<<"\t"<<StsFSCxz[i]<<"\t"<<StsFSCyz[i]<<endl;
    }
}


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
                std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz) 

{
    // Check if there is a value greater than 0 in the Z coordinate
    if (*std::max_element(Z.begin(), Z.end()) > 0 || 
        P1[2] > 0 || P2[2] > 0 || P3[2] > 0) 
    {
        std::cout << "Half-space solution: Z coordinates must be negative!" << std::endl;
    }
    size_t n = X.size();
    std::vector<double> Sxxv(n), Syyv(n), Szzv(n);
    std::vector<double> Sxyv(n), Sxzv(n), Syzv(n);
    std::vector<double> Exxv(n), Eyyv(n), Ezzv(n);
    std::vector<double> Exyv(n), Exzv(n), Eyzv(n);

    std::vector<double> StsFSCxx(n), StsFSCyy(n), StsFSCzz(n);
    std::vector<double> StsFSCxy(n), StsFSCxz(n), StsFSCyz(n);
    std::vector<double> StrFSCxx(n), StrFSCyy(n), StrFSCzz(n);
    std::vector<double> StrFSCxy(n), StrFSCxz(n), StrFSCyz(n);


    TDstressFS(X, Y, Z,
                P1, P2, P3,
                Ss, Ds, Ts,
                mu, lambda_,
                Sxx, Syy, Szz, Sxy, Sxz, Syz,
                Exx, Eyy, Ezz, Exy, Exz, Eyz);
    TDstress_HarFunc(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, mu, lambda_,
                                StsFSCxx, StsFSCyy, StsFSCzz, StsFSCxy, StsFSCxz, StsFSCyz,
                                StrFSCxx, StrFSCyy, StrFSCzz, StrFSCxy, StrFSCxz, StrFSCyz);
    //Calculate image dislocation contribution to displacements
    //  for (size_t i = 0; i < n; ++i) 
    // {
    //     cout << "Observation Point " << i + 1 << ":\n";
    //     cout << "  Stress: ";
    //     cout << "StsFSCxx = " << StsFSCxx[i] << ", StsFSCyy = " << StsFSCyy[i] << ", StsFSCzz = " << StsFSCzz[i] << ", ";
    //     cout << "StsFSCxy = " << StsFSCxy[i] << ", StsFSCxz = " << StsFSCxz[i] << ", StsFSCyz = " << StsFSCyz[i] << "\n";

    //     cout << "--------------------------------------------------\n";
    // }
    
    vector<double> P1_ = P1,P2_ = P2,P3_ = P3;
    P1_[2] = -P1[2];
    P2_[2] = -P2[2];
    P3_[2] = -P3[2];

    TDstressFS(X, Y, Z,
                P1_, P2_, P3_,
                Ss, Ds, Ts,
                mu, lambda_,
                Sxxv, Syyv, Szzv, Sxyv, Sxzv, Syzv,
                Exxv, Eyyv, Ezzv, Exyv, Exzv, Eyzv);
    
    if (P1[2] == 0 && P2[2] == 0 && P3[2] == 0) 
    {
        for (size_t i = 0; i < Sxyv.size(); ++i) 
        {
            Sxzv[i] = -Sxzv[i];  // σ13
            Syzv[i] = -Syzv[i];  // σ23
            Exzv[i] = -Exzv[i];  // ε13
            Eyzv[i] = -Eyzv[i];  // ε23
        }
    }
    for (size_t i = 0; i < Sxx.size(); ++i) 
    {
        Exx[i]=Exx[i]+StrFSCxx[i]+Exxv[i];
        Eyy[i]=Eyy[i]+StrFSCyy[i]+Eyyv[i];
        Ezz[i]=Ezz[i]+StrFSCzz[i]+Ezzv[i];
        Exy[i]=Exy[i]+StrFSCxy[i]+Exyv[i];
        Exz[i]=Exz[i]+StrFSCxz[i]+Exzv[i];
        Eyz[i]=Eyz[i]+StrFSCyz[i]+Eyzv[i];
        Sxx[i]=Sxx[i]+StsFSCxx[i]+Sxxv[i];
        Syy[i]=Syy[i]+StsFSCyy[i]+Syyv[i];
        Szz[i]=Szz[i]+StsFSCzz[i]+Szzv[i];
        Sxy[i]=Sxy[i]+StsFSCxy[i]+Sxyv[i];
        Sxz[i]=Sxz[i]+StsFSCxz[i]+Sxzv[i];
        Syz[i]=Syz[i]+StsFSCyz[i]+Syzv[i];
    }

    // for (size_t i = 0; i < n; ++i) 
    // {
    //     cout << "Observation Point " << i + 1 << ":\n";
    //     cout << "  Stress: ";
    //     cout << "Sxx = " << Sxx[i] << ", Syy = " << Syy[i] << ", Szz = " << Szz[i] << ", ";
    //     cout << "Sxy = " << Sxy[i] << ", Sxz = " << Sxz[i] << ", Syz = " << Syz[i] << "\n";

    //     cout << "--------------------------------------------------\n";
    // }
}

int main()
{
    // Input: 3 observation points
    // vector<double> X = {48771.61733949,-48484.78117283,48775.26818703};
    // vector<double> Y = {30000, 15000, 3000};
    // vector<double> Z = {-13944.10291502,-20994.9909562,-12042.91174556};

    // //Three vertices of a triangle
    // vector<double> P1 = {-49019.8529842,0., -13632.75018015};
    // vector<double> P2 = {-49294.4187163,3000., -14516.85867197};
    // vector<double> P3 = {-48427.27119853,0.,-14642.195744633};
    
    vector<double> X = {29387.75593897,-48489.89772099,-48479.08845632};
    vector<double> Y = {0., 0., 0.};
    vector<double> Z = {-38297.33922684,-37991.33142056,-14039.49925884};

    //Three vertices of a triangle
    vector<double> P1 = {48120.9759287,      0.,         -6947.86435537};
    vector<double> P2 = {49133.9746038,      0.,         -7500.00001313};
    vector<double> P3 = {49133.97459622,     0.,        -6500.};

    // Output: space allocated for each component
    size_t N = X.size();
    vector<double> Sxx(N), Syy(N), Szz(N);
    vector<double> Sxy(N), Sxz(N), Syz(N);
    vector<double> Exx(N), Eyy(N), Ezz(N);
    vector<double> Exy(N), Exz(N), Eyz(N);

    // Dislocation and medium parameters
    double Ss = 1.0, Ds = 0.5, Ts = 0.0;
    double mu = 32038120320, lambda_ = 32038120320;
    //std::cout << "hello"<<endl;
    // Calling the TDstressHS function
    TDstressHS(X, Y, Z, P1, P2, P3,
               Ss, Ds, Ts, mu, lambda_,
               Sxx, Syy, Szz, Sxy, Sxz, Syz,
               Exx, Eyy, Ezz, Exy, Exz, Eyz);
    for (int i = 0; i < N; i++) {
    cout << i << " "
         << Sxx[i] << " " << Syy[i] << " " << Szz[i] << " "
         << Sxy[i] << " " << Sxz[i] << " " << Syz[i] << " "
         << endl;
    }
}







void TDstressFS(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z,
                const std::vector<double>& P1, const std::vector<double>& P2, const std::vector<double>& P3,
                double Ss, double Ds, double Ts, double mu, double lambda_,
                std::vector<double>& Sxx, std::vector<double>& Syy, std::vector<double>& Szz,
                std::vector<double>& Sxy, std::vector<double>& Sxz, std::vector<double>& Syz,
                std::vector<double>& Exx, std::vector<double>& Eyy, std::vector<double>& Ezz,
                std::vector<double>& Exy, std::vector<double>& Exz, std::vector<double>& Eyz);

//
extern "C" void TDstressFS_C(
    const double* X, const double* Y, const double* Z,
	size_t n,
    const double* P1, const double* P2, const double* P3,
    double Ss, double Ds, double Ts,
    double mu, double lambda_,
    bool jud_halfspace, 
    double* stress,
    double* strain)
{
	//printf("n=%d",n);
    // def receivers
    std::vector<double> Xv(X, X + n);
    std::vector<double> Yv(Y, Y + n);
    std::vector<double> Zv(Z, Z + n);

    // for (int i = 0; i < n; i++) 
    // {
    // cout<< "X "<< i <<"  "<<Xv[i]<<endl;
    // }

    // P1, P2, P3 
    std::vector<double> P1v(P1, P1 + 3);
    std::vector<double> P2v(P2, P2 + 3);
    std::vector<double> P3v(P3, P3 + 3);

    // for (int i = 0; i < 3; i++) 
    // {
    // cout<< "P1 "<< i <<"  "<<P1[i]<<endl;
    // }

    // stress and strain
    std::vector<double> Sxxv(n), Syyv(n), Szzv(n);
    std::vector<double> Sxyv(n), Sxzv(n), Syzv(n);
    std::vector<double> Exxv(n), Eyyv(n), Ezzv(n);
    std::vector<double> Exyv(n), Exzv(n), Eyzv(n);

    // calculate the stress
    if (jud_halfspace) {
        TDstressHS(Xv, Yv, Zv, P1v, P2v, P3v, Ss, Ds, Ts, mu, lambda_,
                   Sxxv, Syyv, Szzv, Sxyv, Sxzv, Syzv,
                   Exxv, Eyyv, Ezzv, Exyv, Exzv, Eyzv);
    } else {
        TDstressFS(Xv, Yv, Zv, P1v, P2v, P3v, Ss, Ds, Ts, mu, lambda_,
                   Sxxv, Syyv, Szzv, Sxyv, Sxzv, Syzv,
                   Exxv, Eyyv, Ezzv, Exyv, Exzv, Eyzv);
    }

    // translate into C language
    
	for (size_t i = 0; i < n; ++i) {
        stress[i * 6 + 0] = Sxxv[i]; stress[i * 6 + 1] = Syyv[i]; stress[i * 6 + 2] = Szzv[i];
        stress[i * 6 + 3] = Sxyv[i]; stress[i * 6 + 4] = Sxzv[i]; stress[i * 6 + 5] = Syzv[i];

        strain[i * 6 + 0] = Exxv[i]; strain[i * 6 + 1] = Eyyv[i]; strain[i * 6 + 2] = Ezzv[i];
        strain[i * 6 + 3] = Exyv[i]; strain[i * 6 + 4] = Exzv[i]; strain[i * 6 + 5] = Eyzv[i];
    }
}


extern "C" void TDstressEachSourceAtReceiver_C(
    double x, double y, double z,
    const double* P1_flat, const double* P2_flat, const double* P3_flat,
    size_t n_sources,
    double Ss, double Ds, double Ts,
    double mu, double lambda_,
    bool jud_halfspace,  
    double* stress_out,
    double* strain_out)
{
    std::vector<double> X(1, x), Y(1, y), Z(1, z);
    std::vector<double> sxx(1), syy(1), szz(1), sxy(1), sxz(1), syz(1);
    std::vector<double> exx(1), eyy(1), ezz(1), exy(1), exz(1), eyz(1);

    for (size_t i = 0; i < n_sources; ++i) {
        const double* P1 = P1_flat + i * 3;
        const double* P2 = P2_flat + i * 3;
        const double* P3 = P3_flat + i * 3;

        std::vector<double> P1_vec(P1, P1 + 3);
        std::vector<double> P2_vec(P2, P2 + 3);
        std::vector<double> P3_vec(P3, P3 + 3);

        if (jud_halfspace) {
            TDstressHS(X, Y, Z,
                       P1_vec, P2_vec, P3_vec,
                       Ss, Ds, Ts,
                       mu, lambda_,
                       sxx, syy, szz, sxy, sxz, syz,
                       exx, eyy, ezz, exy, exz, eyz);
        } else {
            TDstressFS(X, Y, Z,
                       P1_vec, P2_vec, P3_vec,
                       Ss, Ds, Ts,
                       mu, lambda_,
                       sxx, syy, szz, sxy, sxz, syz,
                       exx, eyy, ezz, exy, exz, eyz);
        }

        stress_out[i * 6 + 0] = sxx[0]; stress_out[i * 6 + 1] = syy[0]; stress_out[i * 6 + 2] = szz[0];
        stress_out[i * 6 + 3] = sxy[0]; stress_out[i * 6 + 4] = sxz[0]; stress_out[i * 6 + 5] = syz[0];

        strain_out[i * 6 + 0] = exx[0]; strain_out[i * 6 + 1] = eyy[0]; strain_out[i * 6 + 2] = ezz[0];
        strain_out[i * 6 + 3] = exy[0]; strain_out[i * 6 + 4] = exz[0]; strain_out[i * 6 + 5] = eyz[0];
    }
}

