
// This code is modified from source code by Barbot, Sylvain. 
//"Deformation of a Half‐Space from Anelastic Strain Confined in a Tetrahedral Volume." Bulletin of the Seismological Society of America (2018), doi: 10.1785/0120180058..

#include "tetrahedron_stress.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tetra {
namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
inline double powi(double x, int n) {
    switch (n) {
        case 2: return x*x;
        case 3: return x*x*x;
        case 4: { const double y=x*x; return y*y; }
        case 5: { const double y=x*x; return y*y*x; }
        case 6: { const double y=x*x*x; return y*y; }
        case 7: { const double y=x*x*x; return y*y*x; }
        default: return std::pow(x,n);
    }
}

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 operator*(double a, Vec3 b) { return {a*b.x,a*b.y,a*b.z}; }
Vec3 operator/(Vec3 a, double b) { return {a.x/b,a.y/b,a.z/b}; }
double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
double norm(Vec3 a) { return std::sqrt(dot(a,a)); }

using Tensor3 = std::array<std::array<std::array<double,3>,3>,3>;
using Matrix3 = std::array<std::array<double,3>,3>;

Tensor3 greenGradient(const Vec3& x, const Vec3& y, double nu) {
    const double x1=x.x,x2=x.y,x3=x.z,y1=y.x,y2=y.y,y3=y.z;
    const double R1=std::sqrt(powi(x1-y1,2)+powi(x2-y2,2)+powi(x3-y3,2));
    const double R2=std::sqrt(powi(x1-y1,2)+powi(x2-y2,2)+powi(x3+y3,2));
    if (R1 == 0.0) throw std::domain_error("observation coincides with a quadrature source point");
    Tensor3 K{}; // K[source-force k][displacement i][derivative j]
    K[0][0][0] = 1/(16*PI*(1-nu))*(x1-y1)*( -(3-4*nu)/powi(R1,3) -1/powi(R2,3) +(2*powi(R1,2)-3*powi((x1-y1),2))/powi(R1,5) +(3-4*nu)*(2*powi(R2,2)-3*powi((x1-y1),2))/powi(R2,5) -6*y3*x3*(3*powi(R2,2)-5*powi((x1-y1),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) -8*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) +4*(1-2*nu)*(1-nu)*powi((x1-y1),2)/(powi(R2,3)*powi((R2+x3+y3),2)) +8*(1-2*nu)*(1-nu)*powi((x1-y1),2)/(powi(R2,2)*powi((R2+x3+y3),3)) ); // G11d1
    K[0][0][1] = 1/(16*PI*(1-nu))*(x2-y2)*( -(3-4*nu)/powi(R1,3) -1/powi(R2,3) -3*powi((x1-y1),2)/powi(R1,5) -3*(3-4*nu)*powi((x1-y1),2)/powi(R2,5) -6*y3*x3*(powi(R2,2)-5*powi((x1-y1),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) +4*(1-2*nu)*(1-nu)*powi((x1-y1),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G11d2
    K[0][0][2] = 1/(16*PI*(1-nu))*( -(3-4*nu)*(x3-y3)/powi(R1,3) -(x3+y3)/powi(R2,3) -3*powi((x1-y1),2)*(x3-y3)/powi(R1,5) -3*(3-4*nu)*powi((x1-y1),2)*(x3+y3)/powi(R2,5) +2*y3*(powi(R2,2)-3*x3*(x3+y3))/powi(R2,5) -6*y3*powi((x1-y1),2)*(powi(R2,2)-5*x3*(x3+y3))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*(R2+x3+y3)) +4*(1-2*nu)*(1-nu)*powi((x1-y1),2)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G11d3
    K[0][1][0] = 1/(16*PI*(1-nu))*(x2-y2)*( +(powi(R1,2)-3*powi((x1-y1),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*powi((x1-y1),2))/powi(R2,5) -6*y3*x3*(powi(R2,2)-5*powi((x1-y1),2))/powi(R2,7) -4*(1-nu)*(1-2*nu)/(R2*powi((R2+x3+y3),2)) +4*(1-nu)*(1-2*nu)*powi((x1-y1),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G12d1
    K[0][1][1] = 1/(16*PI*(1-nu))*(x1-y1)*( +(powi(R1,2)-3*powi((x2-y2),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*powi((x2-y2),2))/powi(R2,5) -6*y3*x3*(powi(R2,2)-5*powi((x2-y2),2))/powi(R2,7) -4*(1-nu)*(1-2*nu)/(R2*powi((R2+x3+y3),2)) +4*(1-nu)*(1-2*nu)*powi((x2-y2),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G12d2
    K[0][1][2] = 1/(16*PI*(1-nu))*(x1-y1)*(x2-y2)*( -3*(x3-y3)/powi(R1,5) -3*(3-4*nu)*(x3+y3)/powi(R2,5) -6*y3*(powi(R2,2)-5*x3*(x3+y3))/powi(R2,7) +4*(1-2*nu)*(1-nu)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G12d3
    K[0][2][0] = 1/(16*PI*(1-nu))*( +(x3-y3)*(powi(R1,2)-3*powi((x1-y1),2))/powi(R1,5) +(3-4*nu)*(x3-y3)*(powi(R2,2)-3*powi((x1-y1),2))/powi(R2,5) -6*y3*x3*(x3+y3)*(powi(R2,2)-5*powi((x1-y1),2))/powi(R2,7) +4*(1-2*nu)*(1-nu)/(R2*(R2+x3+y3)) -4*(1-2*nu)*(1-nu)*powi((x1-y1),2)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G13d1
    K[0][2][1] = 1/(16*PI*(1-nu))*(x1-y1)*(x2-y2)*( -3*(x3-y3)/powi(R1,5) -3*(3-4*nu)*(x3-y3)/powi(R2,5) +30*y3*x3*(x3+y3)/powi(R2,7) -4*(1-2*nu)*(1-nu)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G13d2
    K[0][2][2] = 1/(16*PI*(1-nu))*(x1-y1)*( +(powi(R1,2)-3*powi((x3-y3),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*(powi(x3,2)-powi(y3,2)))/powi(R2,5) -6*y3*(2*x3+y3)/powi(R2,5) +30*y3*x3*powi((x3+y3),2)/powi(R2,7) -4*(1-2*nu)*(1-nu)/powi(R2,3) ); // G13d3
    K[1][0][0] = 1/(16*PI*(1-nu))*(x2-y2)*( +(powi(R1,2)-3*powi((x1-y1),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*powi((x1-y1),2))/powi(R2,5) -6*y3*x3*(powi(R2,2)-5*powi((x1-y1),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) +4*(1-2*nu)*(1-nu)*powi((x1-y1),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G21d1
    K[1][0][1] = 1/(16*PI*(1-nu))*(x1-y1)*( +(powi(R1,2)-3*powi((x2-y2),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*powi((x2-y2),2))/powi(R2,5) -6*y3*x3*(powi(R2,2)-5*powi((x2-y2),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) +4*(1-2*nu)*(1-nu)*powi((x2-y2),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G21d2
    K[1][0][2] = 1/(16*PI*(1-nu))*(x1-y1)*(x2-y2)*( -3*(x3-y3)/powi(R1,5) -3*(3-4*nu)*(x3+y3)/powi(R2,5) -6*y3*(powi(R2,2)-5*x3*(x3+y3))/powi(R2,7) +4*(1-2*nu)*(1-nu)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G21d3
    K[1][1][0] = 1/(16*PI*(1-nu))*(x1-y1)*( -(3-4*nu)/powi(R1,3) -1/powi(R2,3) -3*powi((x2-y2),2)/powi(R1,5) -3*(3-4*nu)*powi((x2-y2),2)/powi(R2,5) -6*y3*x3*(powi(R2,2)-5*powi((x2-y2),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) +4*(1-2*nu)*(1-nu)*powi((x2-y2),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G22d1
    K[1][1][1] = 1/(16*PI*(1-nu))*(x2-y2)*( -(3-4*nu)/powi(R1,3) -1/powi(R2,3) +(2*powi(R1,2)-3*powi((x2-y2),2))/powi(R1,5) +(3-4*nu)*(2*powi(R2,2)-3*powi((x2-y2),2))/powi(R2,5) -6*y3*x3*(3*powi(R2,2)-5*powi((x2-y2),2))/powi(R2,7) -12*(1-2*nu)*(1-nu)/(R2*powi((R2+x3+y3),2)) +4*(1-2*nu)*(1-nu)*powi((x2-y2),2)*(3*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),3)) ); // G22d2
    K[1][1][2] = 1/(16*PI*(1-nu))*( -(3-4*nu)*(x3-y3)/powi(R1,3) -(x3+y3)/powi(R2,3) -3*powi((x2-y2),2)*(x3-y3)/powi(R1,5) -3*(3-4*nu)*powi((x2-y2),2)*(x3+y3)/powi(R2,5) +2*y3*(powi(R2,2)-3*x3*(x3+y3))/powi(R2,5) -6*y3*powi((x2-y2),2)*(powi(R2,2)-5*x3*(x3+y3))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*(R2+x3+y3)) +4*(1-2*nu)*(1-nu)*powi((x2-y2),2)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G22d3
    K[1][2][0] = 1/(16*PI*(1-nu))*(x1-y1)*(x2-y2)*( -3*(x3-y3)/powi(R1,5) -3*(3-4*nu)*(x3-y3)/powi(R2,5) +30*y3*x3*(x3+y3)/powi(R2,7) -4*(1-2*nu)*(1-nu)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G23d1
    K[1][2][1] = 1/(16*PI*(1-nu))*( +(x3-y3)*(powi(R1,2)-3*powi((x2-y2),2))/powi(R1,5) +(3-4*nu)*(x3-y3)*(powi(R2,2)-3*powi((x2-y2),2))/powi(R2,5) -6*y3*x3*(x3+y3)*(powi(R2,2)-5*powi((x2-y2),2))/powi(R2,7) +4*(1-2*nu)*(1-nu)/(R2*(R2+x3+y3)) -4*(1-2*nu)*(1-nu)*powi((x2-y2),2)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G23d2
    K[1][2][2] = 1/(16*PI*(1-nu))*(x2-y2)*( +(powi(R1,2)-3*powi((x3-y3),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*(powi(x3,2)-powi(y3,2)))/powi(R2,5) -6*y3*(2*x3+y3)/powi(R2,5) +30*y3*x3*powi((x3+y3),2)/powi(R2,7) -4*(1-2*nu)*(1-nu)/powi(R2,3) ); // G23d3
    K[2][0][0] = 1/(16*PI*(1-nu))*( +(x3-y3)*(powi(R1,2)-3*powi((x1-y1),2))/powi(R1,5) +(3-4*nu)*(x3-y3)*(powi(R2,2)-3*powi((x1-y1),2))/powi(R2,5) +6*y3*x3*(x3+y3)*(powi(R2,2)-5*powi((x1-y1),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*(R2+x3+y3)) +4*(1-2*nu)*(1-nu)*powi((x1-y1),2)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G31d1
    K[2][0][1] = 1/(16*PI*(1-nu))*(x1-y1)*(x2-y2)*( -3*(x3-y3)/powi(R1,5) -3*(3-4*nu)*(x3-y3)/powi(R2,5) -30*y3*x3*(x3+y3)/powi(R2,7) +4*(1-2*nu)*(1-nu)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G31d2
    K[2][0][2] = 1/(16*PI*(1-nu))*(x1-y1)*( +(powi(R1,2)-3*powi((x3-y3),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*(powi(x3,2)-powi(y3,2)))/powi(R2,5) +6*y3*(2*x3+y3)/powi(R2,5) -30*y3*x3*powi((x3+y3),2)/powi(R2,7) +4*(1-2*nu)*(1-nu)/powi(R2,3) ); // G31d3
    K[2][1][0] = 1/(16*PI*(1-nu))*(x1-y1)*(x2-y2)*( -3*(x3-y3)/powi(R1,5) -3*(3-4*nu)*(x3-y3)/powi(R2,5) -30*y3*x3*(x3+y3)/powi(R2,7) +4*(1-2*nu)*(1-nu)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G32d1
    K[2][1][1] = 1/(16*PI*(1-nu))*( +(x3-y3)*(powi(R1,2)-3*powi((x2-y2),2))/powi(R1,5) +(3-4*nu)*(x3-y3)*(powi(R2,2)-3*powi((x2-y2),2))/powi(R2,5) +6*y3*x3*(x3+y3)*(powi(R2,2)-5*powi((x2-y2),2))/powi(R2,7) -4*(1-2*nu)*(1-nu)/(R2*(R2+x3+y3)) +4*(1-2*nu)*(1-nu)*powi((x2-y2),2)*(2*R2+x3+y3)/(powi(R2,3)*powi((R2+x3+y3),2)) ); // G32d2
    K[2][1][2] = 1/(16*PI*(1-nu))*(x2-y2)*( +(powi(R1,2)-3*powi((x3-y3),2))/powi(R1,5) +(3-4*nu)*(powi(R2,2)-3*(powi(x3,2)-powi(y3,2)))/powi(R2,5) +6*y3*(2*x3+y3)/powi(R2,5) -30*y3*x3*powi((x3+y3),2)/powi(R2,7) +4*(1-2*nu)*(1-nu)/powi(R2,3) ); // G32d3
    K[2][2][0] = 1/(16*PI*(1-nu))*(x1-y1)*( -(3-4*nu)/powi(R1,3) -(5-12*nu+8*powi(nu,2))/powi(R2,3) -3*powi((x3-y3),2)/powi(R1,5) -30*y3*x3*powi((x3+y3),2)/powi(R2,7) -3*(3-4*nu)*powi((x3+y3),2)/powi(R2,5) +6*y3*x3/powi(R2,5) ); // G33d1
    K[2][2][1] = 1/(16*PI*(1-nu))*(x2-y2)*( -(3-4*nu)/powi(R1,3) -(5-12*nu+8*powi(nu,2))/powi(R2,3) -3*powi((x3-y3),2)/powi(R1,5) -30*y3*x3*powi((x3+y3),2)/powi(R2,7) -3*(3-4*nu)*powi((x3+y3),2)/powi(R2,5) +6*y3*x3/powi(R2,5) ); // G33d2
    K[2][2][2] = 1/(16*PI*(1-nu))*( -(3-4*nu)*(x3-y3)/powi(R1,3) -(5-12*nu+8*powi(nu,2))*(x3+y3)/powi(R2,3) +(x3-y3)*(2*powi(R1,2)-3*powi((x3-y3),2))/powi(R1,5) +6*y3*powi((x3+y3),2)/powi(R2,5) +6*y3*x3*(x3+y3)*(2*powi(R2,2)-5*powi((x3+y3),2))/powi(R2,7) +(3-4*nu)*(x3+y3)*(2*powi(R2,2)-3*powi((x3+y3),2))/powi(R2,5) -2*y3*(powi(R2,2)-3*x3*(x3+y3))/powi(R2,5) ); // G33d3
    return K;
}

struct Face { Vec3 p, q, r, normal; double area; };

std::array<Face,4> makeFaces(const Tetrahedron& t) {
    if(t.A.z<0||t.B.z<0||t.C.z<0||t.D.z<0)
        throw std::invalid_argument("source vertex depth must be nonnegative");
    std::array<Face,4> f{{
        {t.A,t.B,t.C,{},0.0}, {t.B,t.C,t.D,{},0.0},
        {t.C,t.D,t.A,{},0.0}, {t.D,t.A,t.B,{},0.0}}};
    const std::array<Vec3,4> opposite{{t.D,t.A,t.B,t.C}};
    for (std::size_t i=0;i<4;++i) {
        Vec3 c=cross(f[i].r-f[i].p,f[i].q-f[i].p);
        const double twice=norm(c);
        if (!(twice>0.0)) throw std::invalid_argument("degenerate tetrahedron face");
        f[i].area=0.5*twice;
        f[i].normal=c/twice;
        const Vec3 center=(f[i].p+f[i].q+f[i].r)/3.0;
        if (dot(f[i].normal,opposite[i]-center)>0.0) f[i].normal=-1.0*f[i].normal;
    }
    return f;
}

Matrix3 moment(const Eigenstrain& e,double nu) {
    if (!(nu>-1.0 && nu<0.5)) throw std::invalid_argument("Poisson ratio must lie in (-1,0.5)");
    const double lambda=2.0*nu/(1.0-2.0*nu), tr=e.e11+e.e22+e.e33;
    return {{{2*e.e11+lambda*tr,2*e.e12,2*e.e13},
             {2*e.e12,2*e.e22+lambda*tr,2*e.e23},
             {2*e.e13,2*e.e23,2*e.e33+lambda*tr}}};
}

Vec3 mapTriangle(double u,double v,const Face& f) {
    return ((1-u)*(1-v)/4.0)*f.p+((1+u)*(1-v)/4.0)*f.q+((1+v)/2.0)*f.r;
}

void accumulate(Matrix3& grad,const Tensor3& K,Vec3 traction,double weight) {
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        for(int k=0;k<3;++k) grad[i][j]+=weight*K[k][i][j]*traction[k];
}

bool inside(const Vec3& x,const std::array<Face,4>& faces) {
    for(const auto& f:faces) {
        const Vec3 center=(f.p+f.q+f.r)/3.0;
        if (dot(center-x,f.normal)<=0.0) return false; // MATLAB heaviside: boundary is outside
    }
    return true;
}

Stress finish(const Vec3& x,const std::array<Face,4>& faces,const Matrix3& u,
              const Eigenstrain& src,double G,double nu) {
    const double omega=inside(x,faces)?1.0:0.0;
    const double e11=u[0][0]-omega*src.e11;
    const double e12=0.5*(u[0][1]+u[1][0])-omega*src.e12;
    const double e13=0.5*(u[0][2]+u[2][0])-omega*src.e13;
    const double e22=u[1][1]-omega*src.e22;
    const double e23=0.5*(u[1][2]+u[2][1])-omega*src.e23;
    const double e33=u[2][2]-omega*src.e33;
    const double div=e11+e22+e33, lambda=2.0*nu/(1.0-2.0*nu);
    return {2*G*e11+G*lambda*div,2*G*e12,2*G*e13,
            2*G*e22+G*lambda*div,2*G*e23,2*G*e33+G*lambda*div};
}

std::pair<std::vector<double>,std::vector<double>> gaussLegendre(int n) {
    if(n<1) throw std::invalid_argument("Gauss order must be positive");
    std::vector<double>x(n),w(n);
    const int m=(n+1)/2;
    for(int i=0;i<m;++i){
        double z=std::cos(PI*(i+0.75)/(n+0.5)),z1;
        double pp=0.0;
        do { double p1=1,p2=0; for(int j=1;j<=n;++j){double p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;} pp=n*(z*p1-p2)/(z*z-1);z1=z;z=z1-p1/pp; } while(std::abs(z-z1)>1e-15);
        x[i]=-z;x[n-1-i]=z;w[i]=2/((1-z*z)*pp*pp);w[n-1-i]=w[i];
    }
    return {x,w};
}

Vec3 circumcenter(const Tetrahedron&t) {
    const Vec3 ba=t.B-t.A,ca=t.C-t.A,da=t.D-t.A;
    double M[3][4]={{2*ba.x,2*ba.y,2*ba.z,dot(t.B,t.B)-dot(t.A,t.A)},
                    {2*ca.x,2*ca.y,2*ca.z,dot(t.C,t.C)-dot(t.A,t.A)},
                    {2*da.x,2*da.y,2*da.z,dot(t.D,t.D)-dot(t.A,t.A)}};
    for(int c=0;c<3;++c){int p=c;for(int r=c+1;r<3;++r)if(std::abs(M[r][c])>std::abs(M[p][c]))p=r;
        if(std::abs(M[p][c])<1e-14)throw std::invalid_argument("degenerate tetrahedron");
        for(int j=c;j<4;++j)std::swap(M[c][j],M[p][j]);
        for(int r=0;r<3;++r)if(r!=c){double q=M[r][c]/M[c][c];for(int j=c;j<4;++j)M[r][j]-=q*M[c][j];}}
    return {M[0][3]/M[0][0],M[1][3]/M[1][1],M[2][3]/M[2][2]};
}

} // namespace

Stress computeStressTetrahedronGauss(const Vec3& x,const Tetrahedron&t,const Eigenstrain&e,double G,double nu,int order){
    if(x.z<0)throw std::invalid_argument("depth must be nonnegative");
    const auto faces=makeFaces(t);const auto m=moment(e,nu);const auto [q,w]=gaussLegendre(order);Matrix3 grad{};
    for(const auto&f:faces){Vec3 tr{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr[i]+=m[i][j]*f.normal[j];
        for(int iv=0;iv<order;++iv)for(int iu=0;iu<order;++iu){const Vec3 y=mapTriangle(q[iu],q[iv],f);
            accumulate(grad,greenGradient(x,y,nu),tr,w[iu]*w[iv]*(1-q[iv])*f.area/4.0);}}
    return finish(x,faces,grad,e,G,nu);
}

Stress computeStressTetrahedronTanhSinh(const Vec3& x,const Tetrahedron&t,const Eigenstrain&e,double G,double nu,double h,double bound){
    if(x.z<0||h<=0||bound<=0)throw std::invalid_argument("invalid depth or tanh-sinh parameters");
    const auto faces=makeFaces(t);const auto m=moment(e,nu);const int n=static_cast<int>(bound/h);Matrix3 grad{};
    std::vector<double>q(2*n+1),w(2*n+1);for(int k=-n;k<=n;++k){double z=k*h;q[k+n]=std::tanh(0.5*PI*std::sinh(z));w[k+n]=0.5*h*PI*std::cosh(z)/powi(std::cosh(0.5*PI*std::sinh(z)),2);}
    for(const auto&f:faces){Vec3 tr{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr[i]+=m[i][j]*f.normal[j];
        for(std::size_t iv=0;iv<q.size();++iv)for(std::size_t iu=0;iu<q.size();++iu){const Vec3 y=mapTriangle(q[iu],q[iv],f);
            accumulate(grad,greenGradient(x,y,nu),tr,w[iu]*w[iv]*(1-q[iv])*f.area/4.0);}}
    return finish(x,faces,grad,e,G,nu);
}

Stress computeStressTetrahedronMixedQuad(const Vec3&x,const Tetrahedron&t,const Eigenstrain&e,double G,double nu,int N,double h,double bound){
    const Vec3 O=circumcenter(t);const double r=norm(O-t.A);
    return norm(x-O)<r?computeStressTetrahedronTanhSinh(x,t,e,G,nu,h,bound):computeStressTetrahedronGauss(x,t,e,G,nu,N);
}

std::vector<Stress> computeStressTetrahedronMixedQuad(const std::vector<Vec3>&x,const Tetrahedron&t,const Eigenstrain&e,double G,double nu,int N,double h,double bound){
    std::vector<Stress>out;out.reserve(x.size());for(const auto&p:x)out.push_back(computeStressTetrahedronMixedQuad(p,t,e,G,nu,N,h,bound));return out;
}

namespace {
Tetrahedron fromArrays(const std::array<double,3>&A,const std::array<double,3>&B,const std::array<double,3>&C,const std::array<double,3>&D){
    return {{A[0],A[1],A[2]},{B[0],B[1],B[2]},{C[0],C[1],C[2]},{D[0],D[1],D[2]}};
}
template<class Solver>
void matlabInterface(const std::vector<double>&x1,const std::vector<double>&x2,const std::vector<double>&x3,
 const std::array<double,3>&A,const std::array<double,3>&B,const std::array<double,3>&C,const std::array<double,3>&D,
 double e11,double e12,double e13,double e22,double e23,double e33,
 std::vector<double>&s11,std::vector<double>&s12,std::vector<double>&s13,std::vector<double>&s22,std::vector<double>&s23,std::vector<double>&s33,Solver solver){
    if(x1.size()!=x2.size()||x1.size()!=x3.size())throw std::invalid_argument("x1, x2 and x3 must have the same length");
    const auto t=fromArrays(A,B,C,D);const Eigenstrain e{e11,e12,e13,e22,e23,e33};
    s11.resize(x1.size());s12.resize(x1.size());s13.resize(x1.size());s22.resize(x1.size());s23.resize(x1.size());s33.resize(x1.size());
    for(std::size_t i=0;i<x1.size();++i){const Stress s=solver(Vec3{x1[i],x2[i],x3[i]},t,e);
        s11[i]=s.s11;s12[i]=s.s12;s13[i]=s.s13;s22[i]=s.s22;s23[i]=s.s23;s33[i]=s.s33;}
}
}

void computeStressTetrahedronGauss(const std::vector<double>&x1,const std::vector<double>&x2,const std::vector<double>&x3,
 const std::array<double,3>&A,const std::array<double,3>&B,const std::array<double,3>&C,const std::array<double,3>&D,
 double e11,double e12,double e13,double e22,double e23,double e33,double G,double nu,
 std::vector<double>&s11,std::vector<double>&s12,std::vector<double>&s13,std::vector<double>&s22,std::vector<double>&s23,std::vector<double>&s33,int N){
 matlabInterface(x1,x2,x3,A,B,C,D,e11,e12,e13,e22,e23,e33,s11,s12,s13,s22,s23,s33,
  [=](const Vec3&x,const Tetrahedron&t,const Eigenstrain&e){return computeStressTetrahedronGauss(x,t,e,G,nu,N);});}

void computeStressTetrahedronTanhSinh(const std::vector<double>&x1,const std::vector<double>&x2,const std::vector<double>&x3,
 const std::array<double,3>&A,const std::array<double,3>&B,const std::array<double,3>&C,const std::array<double,3>&D,
 double e11,double e12,double e13,double e22,double e23,double e33,double G,double nu,
 std::vector<double>&s11,std::vector<double>&s12,std::vector<double>&s13,std::vector<double>&s22,std::vector<double>&s23,std::vector<double>&s33,double h,double bound){
 matlabInterface(x1,x2,x3,A,B,C,D,e11,e12,e13,e22,e23,e33,s11,s12,s13,s22,s23,s33,
  [=](const Vec3&x,const Tetrahedron&t,const Eigenstrain&e){return computeStressTetrahedronTanhSinh(x,t,e,G,nu,h,bound);});}

void computeStressTetrahedronMixedQuad(const std::vector<double>&x1,const std::vector<double>&x2,const std::vector<double>&x3,
 const std::array<double,3>&A,const std::array<double,3>&B,const std::array<double,3>&C,const std::array<double,3>&D,
 double e11,double e12,double e13,double e22,double e23,double e33,double G,double nu,
 std::vector<double>&s11,std::vector<double>&s12,std::vector<double>&s13,std::vector<double>&s22,std::vector<double>&s23,std::vector<double>&s33,int N,double h,double bound){
 matlabInterface(x1,x2,x3,A,B,C,D,e11,e12,e13,e22,e23,e33,s11,s12,s13,s22,s23,s33,
  [=](const Vec3&x,const Tetrahedron&t,const Eigenstrain&e){return computeStressTetrahedronMixedQuad(x,t,e,G,nu,N,h,bound);});}

} // namespace tetra




#ifdef TETRAHEDRON_STRESS_PYBIND
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
using Array = py::array_t<double,
    py::array::c_style | py::array::forcecast>;

py::array_t<double> mixed_quad_py(
    const Array& observations,
    const Array& vertices,
    const Array& eigenstrain,
    double shear_modulus,
    double poisson_ratio,
    int gauss_order,
    double tanh_precision,
    double tanh_bound)
{
    const auto obs = observations.request();
    const auto verts = vertices.request();
    const auto eig = eigenstrain.request();

    if (obs.ndim != 2 || obs.shape[1] != 3)
        throw py::value_error("observations must have shape (N, 3)");
    if (verts.ndim != 2 || verts.shape[0] != 4 || verts.shape[1] != 3)
        throw py::value_error("vertices must have shape (4, 3)");
    if (eig.ndim != 1 || eig.shape[0] != 6)
        throw py::value_error("eigenstrain must have shape (6,)");
    if (shear_modulus <= 0.0)
        throw py::value_error("shear_modulus must be positive");
    if (!(poisson_ratio > -1.0 && poisson_ratio < 0.5))
        throw py::value_error("poisson_ratio must be in (-1, 0.5)");
    if (gauss_order <= 0 || tanh_precision <= 0.0 || tanh_bound <= 0.0)
        throw py::value_error("quadrature parameters must be positive");

    const auto* x = static_cast<const double*>(obs.ptr);
    const auto* v = static_cast<const double*>(verts.ptr);
    const auto* e = static_cast<const double*>(eig.ptr);
    const std::size_t n = static_cast<std::size_t>(obs.shape[0]);

    std::vector<tetra::Vec3> points;
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        points.push_back({x[3*i], x[3*i+1], x[3*i+2]});

    const tetra::Tetrahedron source{
        {v[0], v[1], v[2]}, {v[3], v[4], v[5]},
        {v[6], v[7], v[8]}, {v[9], v[10], v[11]}
    };
    const tetra::Eigenstrain source_eigenstrain{
        e[0], e[1], e[2], e[3], e[4], e[5]
    };

    std::vector<tetra::Stress> stress;
    {
        py::gil_scoped_release release;
        stress = tetra::computeStressTetrahedronMixedQuad(
            points, source, source_eigenstrain,
            shear_modulus, poisson_ratio,
            gauss_order, tanh_precision, tanh_bound);
    }

    py::array_t<double> result(py::array::ShapeContainer{
        static_cast<py::ssize_t>(n),
        static_cast<py::ssize_t>(6)
    });
    auto out = result.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n; ++i) {
        out(i,0)=stress[i].s11; out(i,1)=stress[i].s12;
        out(i,2)=stress[i].s13; out(i,3)=stress[i].s22;
        out(i,4)=stress[i].s23; out(i,5)=stress[i].s33;
    }
    return result;
}
} // namespace

PYBIND11_MODULE(volume_stress_lib, m)
{
    m.doc() = "Tetrahedral mixed-quadrature half-space stress";
    m.def("computeStressTetrahedronMixedQuad", &mixed_quad_py,
        py::arg("observations"), py::arg("vertices"),
        py::arg("eigenstrain"), py::arg("shear_modulus"),
        py::arg("poisson_ratio"), py::arg("gauss_order")=7,
        py::arg("tanh_precision")=0.01, py::arg("tanh_bound")=3.0,
        "Return [s11,s12,s13,s22,s23,s33] for N observations.");
}
#endif
