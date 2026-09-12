import numpy as np
from numpy.linalg import norm
from math import *
import matplotlib.pyplot as plt

import ctypes

# 
#lib = ctypes.CDLL('src/libang_dis_strain.so')
#lib = ctypes.CDLL('src/Dll1.dll')



# lib.ang_dis_strain.argtypes = [
#     ctypes.POINTER(ctypes.c_double),  # coord_flat
#     ctypes.c_size_t, 
#     ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # alpha, bx, by, bz, nu
#     ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),  # Exx, Eyy, Ezz
#     ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)   # Exy, Exz, Eyz
# ]

def trimodefinder(coord,p1,p2,p3):
    xp=coord[:,1]
    yp=coord[:,2]
    zp=coord[:,0]
    x1,x2,x3=p1[1],p2[1],p3[1]
    y1,y2,y3=p1[2],p2[2],p3[2]
    #for i in range(len(x)):
    a=((xp-x3)*(y2-y3)-(x2-x3)*(yp-y3))/((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))
    b=((x1-x3)*(yp-y3)-(xp-x3)*(y1-y3))/((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))
    c=1.0-a-b
    
    trimode=np.ones(len(xp))
    index1 = np.where((a <= 0) & (b > c) & (c>a))
    trimode[index1]=-1
    index1 = np.where((b <= 0) & (c > a) & (a>b))
    trimode[index1]=-1
    index1 = np.where((c <= 0) & (a > b) & (b>c))
    trimode[index1]=-1
    index1 = np.where((a == 0) & (b >= 0) & (c>=0))
    trimode[index1]=0
    index1 = np.where((a >= 0) & (b == 0) & (c>=0))
    trimode[index1]=0
    index1 = np.where((a >= 0) & (b >=0) & (c==0))
    trimode[index1]=0
    index1 = np.where((trimode == 0) & (zp != 0))
    trimode[index1]=1
    return trimode



def TDSetupS(coord,alpha,bx,by,bz,nu,TriVertex,SideVec):
    A =np.array([[SideVec[2],-SideVec[1]], SideVec[1:3]])
    #print(A)
    
    #Transform coordinates of the calculation points from TDCS into ADCS
    r1 = np.array([coord[:,1]-TriVertex[1],coord[:,2]-TriVertex[2]]).transpose()
    r1=np.dot(r1,A.transpose())
    coord1=np.copy(coord)
    coord1[:,[1,2]]=r1
    #print(coord1)
    
    
    
    #Transform the in-plane slip vector components from TDCS into ADCS
    r2 = np.dot(A,np.array([by,bz]))
    by1=r2[0]
    bz1=r2[1]
    n_pts=len(coord1)
    Exx = np.zeros(len(coord1))
    Eyy = np.zeros(len(coord1))
    Ezz = np.zeros(len(coord1))
    Exy = np.zeros(len(coord1))
    Exz = np.zeros(len(coord1))
    Eyz = np.zeros(len(coord1))
    Exx,Eyy,Ezz,Exy,Exz,Eyz=AngDisStrain(coord1,-np.pi+alpha,bx,by1,bz1,nu)
    # lib.ang_dis_strain(
    #     coord1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     n_pts,
    #     -np.pi+alpha, bx, by, bz, nu,
    #     Exx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     Eyy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     Ezz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     Exy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     Exz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     Eyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    # )
     
    B=np.zeros([3,3])
    B[0][0]=1.0
    B[1:,1:]=A
    
    #print(B[5])
    #E_arr=np.array([[Exx],[],[]])
    #print(u,v,w)
    #return u,v,w
    Exx,Eyy,Ezz,Exy,Exz,Eyz=TensTrans(Exx,Eyy,Ezz,Exy,Exz,Eyz,B)
    return Exx,Eyy,Ezz,Exy,Exz,Eyz
    
    
def TensTrans(Txx1,Tyy1,Tzz1,Txy1,Txz1,Tyz1,A):
    Txx2 = A[0][0]*A[0][0]*Txx1+2*A[0][0]*A[1][0]*Txy1+2*A[0][0]*A[2][0]*Txz1+2*A[1][0]*A[2][0]*Tyz1+\
        A[1][0]*A[1][0]*Tyy1+A[2][0]*A[2][0]*Tzz1;
    
    
    Tyy2 = A[0][1]*A[0][1]*Txx1+2*A[0][1]*A[1][1]*Txy1+2*A[0][1]*A[2][1]*Txz1+2*A[1][1]*A[2][1]*Tyz1+\
        A[1][1]*A[1][1]*Tyy1+A[2][1]*A[2][1]*Tzz1;
    Tzz2 = A[0][2]*A[0][2]*Txx1+2*A[0][2]*A[1][2]*Txy1+2*A[0][2]*A[2][2]*Txz1+2*A[1][2]*A[2][2]*Tyz1+\
        A[1][2]*A[1][2]*Tyy1+A[2][2]*A[2][2]*Tzz1;
    Txy2 = A[0][0]*A[0][1]*Txx1+(A[0][0]*A[1][1]+A[0][1]*A[1][0])*Txy1+(A[0][0]*A[2][1]+
        A[0][1]*A[2][0])*Txz1+(A[2][1]*A[1][0]+A[2][0]*A[1][1])*Tyz1+A[1][1]*A[1][0]*Tyy1+\
        A[2][0]*A[2][1]*Tzz1
    Txz2 = A[0][0]*A[0][2]*Txx1+(A[0][0]*A[1][2]+A[0][2]*A[1][0])*Txy1+(A[0][0]*A[2][2]+
        A[0][2]*A[2][0])*Txz1+(A[2][2]*A[1][0]+A[2][0]*A[1][2])*Tyz1+A[1][2]*A[1][0]*Tyy1+\
        A[2][0]*A[2][2]*Tzz1
    Tyz2 = A[0][1]*A[0][2]*Txx1+(A[0][2]*A[1][1]+A[0][1]*A[1][2])*Txy1+(A[0][2]*A[2][1]+
        A[0][1]*A[2][2])*Txz1+(A[2][1]*A[1][2]+A[2][2]*A[1][1])*Tyz1+A[1][1]*A[1][2]*Tyy1+\
        A[2][1]*A[2][2]*Tzz1
    return Txx2,Tyy2,Tzz2,Txy2,Txz2,Tyz2

    
def AngDisStrain(coord,alpha,bx,by,bz,nu):
    cosA = cos(alpha)
    sinA = sin(alpha)
    x=coord[:,0]
    y=coord[:,1]
    z=coord[:,2]
    eta = y*cosA-z*sinA
    zeta = y*sinA+z*cosA  
    
    x2 = x*x;
    y2 = y*y;
    z2 = z*z;
    
    r2 = x2+y2+z2;
    r=np.sqrt(r2)
    r3=r2*r
    rz=r*(r-z)
    r2z2=r2*(r-z)*(r-z)
    r3z=r3*(r-z)
    
    W=zeta-r
    W2=W*W
    Wr=W*r
    W2r=W2*r
    Wr3=W*r3
    W2r2=W2*r2
    
    C=(r*cosA-z)/Wr
    S=(r*sinA-y)/Wr
    
    
    # Partial derivatives of the Burgers' function
    rFi_rx = (eta/r/(r-zeta)-y/r/(r-z))/4/pi;
    rFi_ry = (x/r/(r-z)-cosA*x/r/(r-zeta))/4/pi;
    rFi_rz = (sinA*x/r/(r-zeta))/4/pi;
    
    Exx = bx*(rFi_rx)+bx/8/pi/(1-nu)*(eta/Wr+eta*x2/W2r2-eta*x2/Wr3+y/rz-
        x2*y/r2z2-x2*y/r3z)-by*x/8/pi/(1-nu)*(((2*nu+1)/Wr+x2/W2r2-x2/Wr3)*cosA+
        (2*nu+1)/rz-x2/r2z2-x2/r3z)+bz*x*sinA/8/pi/(1-nu)*((2*nu+1)/Wr+x2/W2r2-x2/Wr3)
        

    Eyy = by*(rFi_ry)+\
        bx/8/pi/(1-nu)*((1./Wr+S*S-y2/Wr3)*eta+(2*nu+1)*y/rz-y*y*y/r2z2-y*y*y/r3z-2*nu*cosA*S)-\
        by*x/8/pi/(1-nu)*(1./rz-y2/r2z2-y2/r3z+
        (1./Wr+S*S-y2/Wr3)*cosA)+bz*x*sinA/8/pi/(1-nu)*(1./Wr+S*S-y2/Wr3)
        

    Ezz = bz*(rFi_rz)+bx/8/pi/(1-nu)*(eta/W/r+eta*C*C-eta*z2/Wr3+y*z/r3+
        2*nu*sinA*C)-by*x/8/pi/(1-nu)*((1./Wr+C*C-z2/Wr3)*cosA+z/r3)+\
        bz*x*sinA/8/pi/(1-nu)*(1./Wr+C*C-z2/Wr3)
    
    Exy = bx*(rFi_ry)/2+by*(rFi_rx)/2-\
        bx/8/pi/(1-nu)*(x*y2/r2z2-nu*x/rz+x*y2/r3z-nu*x*cosA/Wr+
        eta*x*S/Wr+eta*x*y/Wr3)+\
        by/8/pi/(1-nu)*(x2*y/r2z2-nu*y/rz+x2*y/r3z+nu*cosA*S+
        x2*y*cosA/Wr3+x2*cosA*S/Wr)-\
        bz*sinA/8/pi/(1-nu)*(nu*S+x2*S/Wr+x2*y/Wr3)
    
    Exz = bx*(rFi_rz)/2+bz*(rFi_rx)/2-\
        bx/8/pi/(1-nu)*(-x*y/r3+nu*x*sinA/Wr+eta*x*C/Wr+
        eta*x*z/Wr3)+\
        by/8/pi/(1-nu)*(-x2/r3+nu/r+nu*cosA*C+x2*z*cosA/Wr3+
        x2*cosA*C/Wr)-\
        bz*sinA/8/pi/(1-nu)*(nu*C+x2*C/Wr+x2*z/Wr3)
    
    
    Eyz = by*(rFi_rz)/2+bz*(rFi_ry)/2+\
        bx/8/pi/(1-nu)*(y2/r3-nu/r-nu*cosA*C+nu*sinA*S+eta*sinA*cosA/W2-
        eta*(y*cosA+z*sinA)/W2r+eta*y*z/W2r2-eta*y*z/Wr3)-\
        by*x/8/pi/(1-nu)*(y/r3+sinA*cosA*cosA/W2-cosA*(y*cosA+z*sinA)/
        W2r+y*z*cosA/W2r2-y*z*cosA/Wr3)-\
        bz*x*sinA/8/pi/(1-nu)*(y*z/Wr3-sinA*cosA/W2+(y*cosA+z*sinA)/
        W2r-y*z/W2r2)  
    
    
    return Exx,Eyy,Ezz,Exy,Exz,Eyz


def TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda_):
    X=X.ravel()
    Y=Y.ravel()
    Z=Z.ravel()
    bx = Ts
    by = Ss
    bz = Ds 

    nu = 1/(1+lambda_/mu)/2; 


    Vnorm = np.cross(P2-P1,P3-P1)
    Vnorm = Vnorm/norm(Vnorm)

    eX,eY,eZ = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
    Vstrike = np.cross(eZ,Vnorm)

    if(norm(Vstrike)==0):
        Vstrike = eX*Vnorm[2]

    Vstrike = Vstrike/norm(Vstrike)
    Vdip = np.cross(Vnorm,Vstrike)



    #Transform coordinates from EFCS into TDCS
    p1 = np.zeros(3)
    p2 = np.zeros(3)
    p3 = np.zeros(3)

    At = np.array([Vnorm,Vstrike,Vdip]).transpose()
    data1=np.array([X-P2[0],Y-P2[1],Z-P2[2]]).transpose()
    X1=np.dot(data1,At)

    p1=np.dot(P1-P2,At)
    p3=np.dot(P3-P2,At)

    #Calculate the unit vectors along TD sides in TDCS
    e12 = (p2-p1)/norm(p2-p1)
    e13 = (p3-p1)/norm(p3-p1)
    e23 = (p3-p2)/norm(p3-p2)


    #Calculate the TD angles
    A = acos(np.dot(e12,e13))
    B = acos(np.dot(-e12,e23))
    C = acos(np.dot(e23,e13))

    Trimode=trimodefinder(X1,p1,p2,p3)
    casepLog = Trimode==1
    casenLog = Trimode==-1
    casezLog = Trimode==0

    Xp = X1[casepLog]
    Xn= X1[casenLog]

    if(np.count_nonzero(casepLog)>0):
        #Calculate first angular dislocation contribution
        Exx1Tp,Eyy1Tp,Ezz1Tp,Exy1Tp,Exz1Tp,Eyz1Tp=TDSetupS(Xp,A,bx,by,bz,nu,p1,-e13)
        Exx2Tp,Eyy2Tp,Ezz2Tp,Exy2Tp,Exz2Tp,Eyz2Tp=TDSetupS(Xp,B,bx,by,bz,nu,p2,e12)
        Exx3Tp,Eyy3Tp,Ezz3Tp,Exy3Tp,Exz3Tp,Eyz3Tp=TDSetupS(Xp,C,bx,by,bz,nu,p3,e23)


    if(np.count_nonzero(casenLog)>0):
        #print('!!!!!!!!!!!!!',np.count_nonzero(casenLog))
        Exx1Tn,Eyy1Tn,Ezz1Tn,Exy1Tn,Exz1Tn,Eyz1Tn=TDSetupS(Xn,A,bx,by,bz,nu,p1,e13)
        Exx2Tn,Eyy2Tn,Ezz2Tn,Exy2Tn,Exz2Tn,Eyz2Tn=TDSetupS(Xn,B,bx,by,bz,nu,p2,-e12)
        Exx3Tn,Eyy3Tn,Ezz3Tn,Exy3Tn,Exz3Tn,Eyz3Tn=TDSetupS(Xn,C,bx,by,bz,nu,p3,-e23)
    #print(Exx1Tn.shape,Exx2Tn.shape,Exx3Tn.shape)



    exx=np.zeros(len(Trimode))
    eyy=np.zeros(len(Trimode))
    ezz=np.zeros(len(Trimode))
    exy=np.zeros(len(Trimode))
    exz=np.zeros(len(Trimode))
    eyz=np.zeros(len(Trimode))

    if(np.count_nonzero(casepLog)>0):
        exx[casepLog]=Exx1Tp+Exx2Tp+Exx3Tp
        eyy[casepLog]=Eyy1Tp+Eyy2Tp+Eyy3Tp
        ezz[casepLog]=Ezz1Tp+Ezz2Tp+Ezz3Tp
        exy[casepLog]=Exy1Tp+Exy2Tp+Exy3Tp
        exz[casepLog]=Exz1Tp+Exz2Tp+Exz3Tp
        eyz[casepLog]=Eyz1Tp+Eyz2Tp+Eyz3Tp

    #print(casepLog.shape,casenLog.shape,exx.shape,Exx1Tn.shape,X.shape)
    if(np.count_nonzero(casenLog)>0):
        exx[casenLog]=Exx1Tn+Exx2Tn+Exx3Tn
        eyy[casenLog]=Eyy1Tn+Eyy2Tn+Eyy3Tn
        ezz[casenLog]=Ezz1Tn+Ezz2Tn+Ezz3Tn
        exy[casenLog]=Exy1Tn+Exy2Tn+Exy3Tn
        exz[casenLog]=Exz1Tn+Exz2Tn+Exz3Tn
        eyz[casenLog]=Eyz1Tn+Eyz2Tn+Eyz3Tn
    
    if(np.count_nonzero(casezLog)>0):
        exx[casezLog]=np.nan
        eyy[casezLog]=np.nan
        ezz[casezLog]=np.nan
        exy[casezLog]=np.nan
        exz[casezLog]=np.nan
        eyz[casezLog]=np.nan

    Exx,Eyy,Ezz,Exy,Exz,Eyz = TensTrans(exx,eyy,ezz,exy,exz,eyz,np.array([Vnorm,Vstrike,Vdip]));


    # Calculate the stress tensor components in EFCS
    Sxx = 2*mu*Exx+lambda_*(Exx+Eyy+Ezz)
    Syy = 2*mu*Eyy+lambda_*(Exx+Eyy+Ezz)
    Szz = 2*mu*Ezz+lambda_*(Exx+Eyy+Ezz)
    Sxy = 2*mu*Exy
    Sxz = 2*mu*Exz
    Syz = 2*mu*Eyz
    
    Strain = np.array([Exx,Eyy,Ezz,Exy,Exz,Eyz])
    Stress = np.array([Sxx,Syy,Szz,Sxy,Sxz,Syz])
    
    return Stress,Strain


def AngDisStrainFSC(y1,y2,y3,beta,b1,b2,b3,nu,a):
    sinB = sin(beta)
    cosB = cos(beta)
    cotB = 1.0/tan(beta)
    y3b = y3+2*a
    z1b = y1*cosB+y3b*sinB
    z3b = -y1*sinB+y3b*cosB
    rb2 = y1**2+y2**2+y3b**2
    rb = np.sqrt(rb2)
    
    W1 = rb*cosB+y3b
    W2 = cosB+a/rb
    W3 = cosB+y3b/rb
    W4 = nu+a/rb
    W5 = 2*nu+a/rb
    W6 = rb+y3b
    W7 = rb+z3b
    W8 = y3+a
    W9 = 1+a/rb/cosB
    N1 = 1-2*nu
    
    #print('rb2 ',rb2)
    #print('W ',W1,W2,W3,W4,W5,W6,W7,W8,W9)
    #print('b  ',b1,b2,b3)
    # Partial derivatives of the Burgers' function
    rFib_ry2 = z1b/rb/(rb+z3b)-y1/rb/(rb+y3b) # y2 = x in ADCS
    rFib_ry1 = y2/rb/(rb+y3b)-cosB*y2/rb/(rb+z3b) # y1 =y in ADCS
    rFib_ry3 = -sinB*y2/rb/(rb+z3b) # y3 = z in ADCS
    #print('y1,z3b ',y1,z3b)
    #print('rFib_ry ',rFib_ry1,rFib_ry2,rFib_ry3)
    
    # print('v11_',v11_)
    v11 = b1*(1/4*((-2+2*nu)*N1*rFib_ry1*cotB**2-N1*y2/W6**2*((1-W5)*cotB-\
        y1/W6*W4)/rb*y1+N1*y2/W6*(a/rb**3*y1*cotB-1.0/W6*W4+y1**2.0/\
        W6**2*W4/rb+y1**2/W6*a/rb**3)-N1*y2*cosB*cotB/W7**2*W2*(y1/\
        rb-sinB)-N1*y2*cosB*cotB/W7*a/rb**3*y1-3*a*y2*W8*cotB/rb**5.*\
        y1-y2*W8/rb**3/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1-y2*W8/\
        rb2/W6**2*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1+y2*W8/rb/W6*\
        (1/W6*W5-y1**2/W6**2*W5/rb-y1**2/W6*a/rb**3+a/rb2-2*a*y1**\
        2/rb2**2)-y2*W8/rb**3/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+\
        (2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)*y1-y2*W8/rb/\
        W7**2*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*\
        cosB)-a*y3b*cosB*cotB/rb2)*(y1/rb-sinB)+y2*W8/rb/W7*(-cosB/\
        W7**2*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)*(y1/\
        rb-sinB)+cosB/W7*(1/rb*cosB*y1*(N1*cosB-a/rb)*cotB+W1*a/rb**\
        3*y1*cotB+(2-2*nu)*(1./rb*sinB*y1-1)*cosB)+2*a*y3b*cosB*cotB/\
        rb2**2.*y1))/pi/(1-nu))+\
        b2*(1./4*(N1*(((2-2*nu)*cotB**2+nu)/rb*y1/W6-((2-2*nu)*cotB**2+1)*\
        cosB*(y1/rb-sinB)/W7)-N1/W6**2.*(-N1*y1*cotB+nu*y3b-a+a*y1*\
        cotB/rb+y1**2./W6*W4)/rb*y1+N1/W6*(-N1*cotB+a*cotB/rb-a*\
        y1**2*cotB/rb**3+2.*y1/W6*W4-y1**3./W6**2.*W4/rb-y1**3./W6*a/\
        rb**3)+N1*cotB/W7**2.*(z1b*cosB-a*(rb*sinB-y1)/rb/cosB)*(y1/\
        rb-sinB)-N1*cotB/W7*(cosB**2-a*(1./rb*sinB*y1-1)/rb/cosB+a*\
        (rb*sinB-y1)/rb**3./cosB*y1)-a*W8*cotB/rb**3+3*a*y1**2.*W8*\
        cotB/rb**5-W8/W6**2.*(2*nu+1./rb*(N1*y1*cotB+a)-y1**2./rb/W6*\
        W5-a*y1**2./rb**3)/rb*y1+W8/W6*(-1./rb**3.*(N1*y1*cotB+a)*y1+\
        1./rb*N1*cotB-2.*y1/rb/W6*W5+y1**3./rb**3/W6*W5+y1**3./rb2/\
        W6**2.*W5+y1**3./rb2**2./W6*a-2*a/rb**3.*y1+3*a*y1**3./rb**5)-W8*\
        cotB/W7**2.*(-cosB*sinB+a*y1*y3b/rb**3./cosB+(rb*sinB-y1)/rb*\
        ((2-2*nu)*cosB-W1/W7*W9))*(y1/rb-sinB)+W8*cotB/W7*(a*y3b/\
        rb**3./cosB-3*a*y1**2.*y3b/rb**5./cosB+(1./rb*sinB*y1-1)/rb*\
        ((2-2*nu)*cosB-W1/W7*W9)-(rb*sinB-y1)/rb**3.*((2-2*nu)*cosB-W1/\
        W7*W9)*y1+(rb*sinB-y1)/rb*(-1./rb*cosB*y1/W7*W9+W1/W7**2.*\
        W9*(y1/rb-sinB)+W1/W7*a/rb**3./cosB*y1)))/pi/(1-nu))+\
        b3*(1/4*(N1*(-y2/W6**2.*(1+a/rb)/rb*y1-y2/W6*a/rb**3.*y1+y2*\
        cosB/W7**2.*W2*(y1/rb-sinB)+y2*cosB/W7*a/rb**3.*y1)+y2*W8/\
        rb**3.*(a/rb2+1./W6)*y1-y2*W8/rb*(-2*a/rb2**2.*y1-1./W6**2./\
        rb*y1)-y2*W8*cosB/rb**3./W7*(W1/W7*W2+a*y3b/rb2)*y1-y2*W8*\
        cosB/rb/W7**2.*(W1/W7*W2+a*y3b/rb2)*(y1/rb-sinB)+y2*W8*\
        cosB/rb/W7*(1./rb*cosB*y1/W7*W2-W1/W7**2.*W2*(y1/rb-sinB)-\
        W1/W7*a/rb**3.*y1-2*a*y3b/rb2**2.*y1))/pi/(1-nu))
    #print('v11_ ',v11)
    v22 = b1*(1/4*(N1*(((2-2*nu)*cotB**2-nu)/rb*y2/W6-((2-2*nu)*cotB**2+1-\
        2*nu)*cosB/rb*y2/W7)+N1/W6**2.*(y1*cotB*(1-W5)+nu*y3b-a+y2**\
        2./W6*W4)/rb*y2-N1/W6*(a*y1*cotB/rb**3.*y2+2.*y2/W6*W4-y2**\
        3./W6**2.*W4/rb-y2**3./W6*a/rb**3)+N1*z1b*cotB/W7**2.*W2/rb*\
        y2+N1*z1b*cotB/W7*a/rb**3.*y2+3*a*y2*W8*cotB/rb**5.*y1-W8/\
        W6**2.*(-2*nu+1./rb*(N1*y1*cotB-a)+y2**2./rb/W6*W5+a*y2**2./\
        rb**3)/rb*y2+W8/W6*(-1./rb**3.*(N1*y1*cotB-a)*y2+2.*y2/rb/\
        W6*W5-y2**3./rb**3./W6*W5-y2**3./rb2/W6**2.*W5-y2**3./rb2**2./W6*\
        a+2*a/rb**3.*y2-3*a*y2**3./rb**5)-W8/W7**2.*(cosB**2-1./rb*(N1*\
        z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**3-1./rb/W7*(y2**2*cosB**2-\
        a*z1b*cotB/rb*W1))/rb*y2+W8/W7*(1./rb**3.*(N1*z1b*cotB+a*\
        cosB)*y2-3*a*y3b*z1b*cotB/rb**5.*y2+1./rb**3./W7*(y2**2*cosB**2-\
        a*z1b*cotB/rb*W1)*y2+1./rb2/W7**2.*(y2**2*cosB**2-a*z1b*cotB/\
        rb*W1)*y2-1./rb/W7*(2.*y2*cosB**2+a*z1b*cotB/rb**3.*W1*y2-a*\
        z1b*cotB/rb2*cosB*y2)))/pi/(1-nu))+\
        b2*(1./4*((2-2*nu)*N1*rFib_ry2*cotB**2+N1/W6*((W5-1)*cotB+y1/W6*\
        W4)-N1*y2**2./W6**2.*((W5-1)*cotB+y1/W6*W4)/rb+N1*y2/W6*(-a/\
        rb**3.*y2*cotB-y1/W6**2.*W4/rb*y2-y2/W6*a/rb**3.*y1)-N1*cotB/\
        W7*W9+N1*y2**2*cotB/W7**2.*W9/rb+N1*y2**2*cotB/W7*a/rb**3./\
        cosB-a*W8*cotB/rb**3+3*a*y2**2.*W8*cotB/rb**5+W8/rb/W6*(N1*\
        cotB-2*nu*y1/W6-a*y1/rb*(1./rb+1./W6))-y2**2.*W8/rb**3./W6*\
        (N1*cotB-2*nu*y1/W6-a*y1/rb*(1./rb+1./W6))-y2**2.*W8/rb2/W6**\
        2.*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1./rb+1./W6))+y2*W8/rb/W6*\
        (2*nu*y1/W6**2./rb*y2+a*y1/rb**3.*(1./rb+1./W6)*y2-a*y1/rb*\
        (-1./rb**3.*y2-1./W6**2./rb*y2))+W8*cotB/rb/W7*((-2+2*nu)*cosB+\
        W1/W7*W9+a*y3b/rb2/cosB)-y2**2.*W8*cotB/rb**3./W7*((-2+2*nu)*\
        cosB+W1/W7*W9+a*y3b/rb2/cosB)-y2**2.*W8*cotB/rb2/W7**2.*((-2+\
        2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)+y2*W8*cotB/rb/W7*(1./\
        rb*cosB*y2/W7*W9-W1/W7**2.*W9/rb*y2-W1/W7*a/rb**3./cosB*y2-\
        2*a*y3b/rb2**2./cosB*y2))/pi/(1-nu))+\
        b3*(1/4*(N1*(-sinB/rb*y2/W7+y2/W6**2.*(1+a/rb)/rb*y1+y2/W6*\
        a/rb**3.*y1-z1b/W7**2.*W2/rb*y2-z1b/W7*a/rb**3.*y2)-y2*W8/\
        rb**3.*(a/rb2+1./W6)*y1+y1*W8/rb*(-2*a/rb2**2.*y2-1./W6**2./\
        rb*y2)+W8/W7**2.*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1./\
        rb/W7*(y2**2*cosB*sinB-a*z1b/rb*W1))/rb*y2-W8/W7*(sinB*a/\
        rb**3.*y2-z1b/rb**3.*(1+a*y3b/rb2)*y2-2.*z1b/rb**5*a*y3b*y2+\
        1./rb**3./W7*(y2**2*cosB*sinB-a*z1b/rb*W1)*y2+1./rb2/W7**2.*\
        (y2**2*cosB*sinB-a*z1b/rb*W1)*y2-1./rb/W7*(2.*y2*cosB*sinB+a*\
        z1b/rb**3.*W1*y2-a*z1b/rb2*cosB*y2)))/pi/(1-nu))
    
    v33 = b1*(1/4*((2-2*nu)*(N1*rFib_ry3*cotB-y2/W6**2.*W5*(y3b/rb+1)-\
        1/2.*y2/W6*a/rb**3*2.*y3b+y2*cosB/W7**2.*W2*W3+1/2.*y2*cosB/W7*\
        a/rb**3*2.*y3b)+y2/rb*(2*nu/W6+a/rb2)-1/2.*y2*W8/rb**3.*(2*\
        nu/W6+a/rb2)*2.*y3b+y2*W8/rb*(-2*nu/W6**2.*(y3b/rb+1)-a/\
        rb2**2*2.*y3b)+y2*cosB/rb/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-\
        1/2.*y2*W8*cosB/rb**3./W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)*2.*\
        y3b-y2*W8*cosB/rb/W7**2.*(1-2*nu-W1/W7*W2-a*y3b/rb2)*W3+y2*\
        W8*cosB/rb/W7*(-(cosB*y3b/rb+1)/W7*W2+W1/W7**2.*W2*W3+1/2.*\
        W1/W7*a/rb**3*2.*y3b-a/rb2+a*y3b/rb2**2*2.*y3b))/pi/(1-nu))+\
        b2*(1/4*((-2+2*nu)*N1*cotB*((y3b/rb+1)/W6-cosB*W3/W7)+(2-2*nu)*\
        y1/W6**2.*W5*(y3b/rb+1)+1/2.*(2-2*nu)*y1/W6*a/rb**3*2.*y3b+(2-\
        2*nu)*sinB/W7*W2-(2-2*nu)*z1b/W7**2.*W2*W3-1/2.*(2-2*nu)*z1b/\
        W7*a/rb**3*2.*y3b+1./rb*(N1*cotB-2*nu*y1/W6-a*y1/rb2)-1/2.*\
        W8/rb**3.*(N1*cotB-2*nu*y1/W6-a*y1/rb2)*2.*y3b+W8/rb*(2*nu*\
        y1/W6**2.*(y3b/rb+1)+a*y1/rb2**2*2.*y3b)-1./W7*(cosB*sinB+W1*\
        cotB/rb*((2-2*nu)*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*\
        W1/rb/W7))+W8/W7**2.*(cosB*sinB+W1*cotB/rb*((2-2*nu)*cosB-W1/\
        W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*W3-W8/W7*((cosB*\
        y3b/rb+1)*cotB/rb*((2-2*nu)*cosB-W1/W7)-1/2.*W1*cotB/rb**3.*\
        ((2-2*nu)*cosB-W1/W7)*2.*y3b+W1*cotB/rb*(-(cosB*y3b/rb+1)/W7+\
        W1/W7**2.*W3)-1/2*a/rb**3.*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7)*\
        2.*y3b+a/rb*(-z1b/rb2-y3b*sinB/rb2+y3b*z1b/rb2**2*2.*y3b-\
        sinB*W1/rb/W7-z1b*(cosB*y3b/rb+1)/rb/W7+1/2.*z1b*W1/rb**3./\
        W7*2.*y3b+z1b*W1/rb/W7**2.*W3)))/pi/(1-nu))+\
        b3*(1/4*((2-2*nu)*rFib_ry3-(2-2*nu)*y2*sinB/W7**2.*W2*W3-1/2.*\
        (2-2*nu)*y2*sinB/W7*a/rb**3*2.*y3b+y2*sinB/rb/W7*(1+W1/W7*\
        W2+a*y3b/rb2)-1/2.*y2*W8*sinB/rb**3./W7*(1+W1/W7*W2+a*y3b/\
        rb2)*2.*y3b-y2*W8*sinB/rb/W7**2.*(1+W1/W7*W2+a*y3b/rb2)*W3+\
        y2*W8*sinB/rb/W7*((cosB*y3b/rb+1)/W7*W2-W1/W7**2.*W2*W3-\
        1/2.*W1/W7*a/rb**3*2.*y3b+a/rb2-a*y3b/rb2**2*2*y3b))/pi/(1-nu))
    
    
    #print('v33 ',v33)
    v12 = b1/2*(1/4*((-2+2*nu)*N1*rFib_ry2*cotB**2+N1/W6*((1-W5)*cotB-y1/\
        W6*W4)-N1*y2**2./W6**2.*((1-W5)*cotB-y1/W6*W4)/rb+N1*y2/W6*\
        (a/rb**3.*y2*cotB+y1/W6**2.*W4/rb*y2+y2/W6*a/rb**3.*y1)+N1*\
        cosB*cotB/W7*W2-N1*y2**2*cosB*cotB/W7**2.*W2/rb-N1*y2**2*cosB*\
        cotB/W7*a/rb**3+a*W8*cotB/rb**3-3*a*y2**2.*W8*cotB/rb**5+W8/\
        rb/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)-y2**2.*W8/rb**3./W6*(-N1*\
        cotB+y1/W6*W5+a*y1/rb2)-y2**2.*W8/rb2/W6**2.*(-N1*cotB+y1/\
        W6*W5+a*y1/rb2)+y2*W8/rb/W6*(-y1/W6**2.*W5/rb*y2-y2/W6*\
        a/rb**3.*y1-2*a*y1/rb2**2.*y2)+W8/rb/W7*(cosB/W7*(W1*(N1*\
        cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/\
        rb2)-y2**2.*W8/rb**3./W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-\
        2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)-y2**2.*W8/rb2/\
        W7**2.*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*\
        cosB)-a*y3b*cosB*cotB/rb2)+y2*W8/rb/W7*(-cosB/W7**2.*(W1*\
        (N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)/rb*y2+cosB/\
        W7*(1./rb*cosB*y2*(N1*cosB-a/rb)*cotB+W1*a/rb**3.*y2*cotB+(2-2*\
        nu)/rb*sinB*y2*cosB)+2*a*y3b*cosB*cotB/rb2**2.*y2))/pi/(1-nu))+\
        b2/2*(1/4*(N1*(((2-2*nu)*cotB**2+nu)/rb*y2/W6-((2-2*nu)*cotB**2+1)*\
        cosB/rb*y2/W7)-N1/W6**2.*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/rb+\
        y1**2./W6*W4)/rb*y2+N1/W6*(-a*y1*cotB/rb**3.*y2-y1**2./W6**\
        2.*W4/rb*y2-y1**2./W6*a/rb**3.*y2)+N1*cotB/W7**2.*(z1b*cosB-a*\
        (rb*sinB-y1)/rb/cosB)/rb*y2-N1*cotB/W7*(-a/rb2*sinB*y2/\
        cosB+a*(rb*sinB-y1)/rb**3./cosB*y2)+3*a*y2*W8*cotB/rb**5.*y1-\
        W8/W6**2.*(2*nu+1./rb*(N1*y1*cotB+a)-y1**2./rb/W6*W5-a*y1**2./\
        rb**3)/rb*y2+W8/W6*(-1./rb**3.*(N1*y1*cotB+a)*y2+y1**2./rb**\
        3./W6*W5*y2+y1**2./rb2/W6**2.*W5*y2+y1**2./rb2**2./W6*a*y2+3*\
        a*y1**2./rb**5.*y2)-W8*cotB/W7**2.*(-cosB*sinB+a*y1*y3b/rb**3./\
        cosB+(rb*sinB-y1)/rb*((2-2*nu)*cosB-W1/W7*W9))/rb*y2+W8*cotB/\
        W7*(-3*a*y1*y3b/rb**5./cosB*y2+1./rb2*sinB*y2*((2-2*nu)*cosB-\
        W1/W7*W9)-(rb*sinB-y1)/rb**3.*((2-2*nu)*cosB-W1/W7*W9)*y2+(rb*\
        sinB-y1)/rb*(-1./rb*cosB*y2/W7*W9+W1/W7**2.*W9/rb*y2+W1/W7*\
        a/rb**3./cosB*y2)))/pi/(1-nu))+\
        b3/2*(1/4*(N1*(1./W6*(1+a/rb)-y2**2./W6**2.*(1+a/rb)/rb-y2**2./\
        W6*a/rb**3-cosB/W7*W2+y2**2*cosB/W7**2.*W2/rb+y2**2*cosB/W7*\
        a/rb**3)-W8/rb*(a/rb2+1./W6)+y2**2.*W8/rb**3.*(a/rb2+1./W6)-\
        y2*W8/rb*(-2*a/rb2**2.*y2-1./W6**2./rb*y2)+W8*cosB/rb/W7*\
        (W1/W7*W2+a*y3b/rb2)-y2**2.*W8*cosB/rb**3./W7*(W1/W7*W2+a*\
        y3b/rb2)-y2**2.*W8*cosB/rb2/W7**2.*(W1/W7*W2+a*y3b/rb2)+y2*\
        W8*cosB/rb/W7*(1./rb*cosB*y2/W7*W2-W1/W7**2.*W2/rb*y2-W1/\
        W7*a/rb**3.*y2-2*a*y3b/rb2**2.*y2))/pi/(1-nu))+\
        b1/2*(1/4*(N1*(((2-2*nu)*cotB**2-nu)/rb*y1/W6-((2-2*nu)*cotB**2+1-\
        2*nu)*cosB*(y1/rb-sinB)/W7)+N1/W6**2.*(y1*cotB*(1-W5)+nu*y3b-\
        a+y2**2./W6*W4)/rb*y1-N1/W6*((1-W5)*cotB+a*y1**2*cotB/rb**3-\
        y2**2./W6**2.*W4/rb*y1-y2**2./W6*a/rb**3.*y1)-N1*cosB*cotB/W7*\
        W2+N1*z1b*cotB/W7**2.*W2*(y1/rb-sinB)+N1*z1b*cotB/W7*a/rb**\
        3.*y1-a*W8*cotB/rb**3+3*a*y1**2.*W8*cotB/rb**5-W8/W6**2.*(-2*\
        nu+1./rb*(N1*y1*cotB-a)+y2**2./rb/W6*W5+a*y2**2./rb**3)/rb*\
        y1+W8/W6*(-1./rb**3.*(N1*y1*cotB-a)*y1+1./rb*N1*cotB-y2**2./\
        rb**3./W6*W5*y1-y2**2./rb2/W6**2.*W5*y1-y2**2./rb2**2./W6*a*y1-\
        3*a*y2**2./rb**5.*y1)-W8/W7**2.*(cosB**2-1./rb*(N1*z1b*cotB+a*\
        cosB)+a*y3b*z1b*cotB/rb**3-1./rb/W7*(y2**2*cosB**2-a*z1b*cotB/\
        rb*W1))*(y1/rb-sinB)+W8/W7*(1./rb**3.*(N1*z1b*cotB+a*cosB)*\
        y1-1./rb*N1*cosB*cotB+a*y3b*cosB*cotB/rb**3-3*a*y3b*z1b*cotB/\
        rb**5.*y1+1./rb**3./W7*(y2**2*cosB**2-a*z1b*cotB/rb*W1)*y1+1./\
        rb/W7**2.*(y2**2*cosB**2-a*z1b*cotB/rb*W1)*(y1/rb-sinB)-1./rb/\
        W7*(-a*cosB*cotB/rb*W1+a*z1b*cotB/rb**3.*W1*y1-a*z1b*cotB/\
        rb2*cosB*y1)))/pi/(1-nu))+\
        b2/2*(1/4*((2-2*nu)*N1*rFib_ry1*cotB**2-N1*y2/W6**2.*((W5-1)*cotB+\
        y1/W6*W4)/rb*y1+N1*y2/W6*(-a/rb**3*y1*cotB+1/W6*W4-y1**\
        2/W6**2*W4/rb-y1**2/W6*a/rb**3)+N1*y2*cotB/W7**2*W9*(y1/\
        rb-sinB)+N1*y2*cotB/W7*a/rb**3/cosB*y1+3*a*y2*W8*cotB/rb**\
        5*y1-y2*W8/rb**3/W6*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1./\
        W6))*y1-y2*W8/rb2/W6**2.*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1./\
        rb+1/W6))*y1+y2*W8/rb/W6*(-2*nu/W6+2*nu*y1**2/W6**2/rb-a/\
        rb*(1/rb+1/W6)+a*y1**2/rb**3.*(1/rb+1/W6)-a*y1/rb*(-1./\
        rb**3*y1-1/W6**2/rb*y1))-y2*W8*cotB/rb**3/W7*((-2+2*nu)*\
        cosB+W1/W7*W9+a*y3b/rb2/cosB)*y1-y2*W8*cotB/rb/W7**2.*((-2+\
        2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)*(y1/rb-sinB)+y2*W8*\
        cotB/rb/W7*(1/rb*cosB*y1/W7*W9-W1/W7**2*W9*(y1/rb-sinB)-\
        W1/W7*a/rb**3/cosB*y1-2*a*y3b/rb2**2/cosB*y1))/pi/(1-nu))+\
        b3/2*(1/4*(N1*(-sinB*(y1/rb-sinB)/W7-1/W6*(1+a/rb)+y1**2/W6**\
        2.*(1+a/rb)/rb+y1**2/W6*a/rb**3+cosB/W7*W2-z1b/W7**2*W2*\
        (y1/rb-sinB)-z1b/W7*a/rb**3*y1)+W8/rb*(a/rb2+1/W6)-y1**2.*\
        W8/rb**3.*(a/rb2+1/W6)+y1*W8/rb*(-2*a/rb2**2*y1-1/W6**2./\
        rb*y1)+W8/W7**2.*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1./\
        rb/W7*(y2**2*cosB*sinB-a*z1b/rb*W1))*(y1/rb-sinB)-W8/W7*\
        (sinB*a/rb**3*y1+cosB/rb*(1+a*y3b/rb2)-z1b/rb**3.*(1+a*y3b/\
        rb2)*y1-2*z1b/rb**5*a*y3b*y1+1/rb**3/W7*(y2**2*cosB*sinB-a*\
        z1b/rb*W1)*y1+1/rb/W7**2.*(y2**2*cosB*sinB-a*z1b/rb*W1)*\
        (y1/rb-sinB)-1/rb/W7*(-a*cosB/rb*W1+a*z1b/rb**3*W1*y1-a*\
        z1b/rb2*cosB*y1)))/pi/(1-nu))
    
    #print('v12_ ',v12)
    v13 = b1/2*(1/4*((-2+2*nu)*N1*rFib_ry3*cotB**2-N1*y2/W6**2.*((1-W5)*\
        cotB-y1/W6*W4)*(y3b/rb+1)+N1*y2/W6*(1/2*a/rb**3*2*y3b*cotB+\
        y1/W6**2*W4*(y3b/rb+1)+1/2*y1/W6*a/rb**3*2*y3b)-N1*y2*cosB*\
        cotB/W7**2*W2*W3-1/2*N1*y2*cosB*cotB/W7*a/rb**3*2*y3b+a/\
        rb**3*y2*cotB-3./2*a*y2*W8*cotB/rb**5*2*y3b+y2/rb/W6*(-N1*\
        cotB+y1/W6*W5+a*y1/rb2)-1/2*y2*W8/rb**3./W6*(-N1*cotB+y1/\
        W6*W5+a*y1/rb2)*2*y3b-y2*W8/rb/W6**2.*(-N1*cotB+y1/W6*W5+\
        a*y1/rb2)*(y3b/rb+1)+y2*W8/rb/W6*(-y1/W6**2*W5*(y3b/rb+\
        1)-1/2*y1/W6*a/rb**3*2*y3b-a*y1/rb2**2*2*y3b)+y2/rb/W7*\
        (cosB/W7*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-\
        a*y3b*cosB*cotB/rb2)-1/2*y2*W8/rb**3./W7*(cosB/W7*(W1*(N1*\
        cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/\
        rb2)*2*y3b-y2*W8/rb/W7**2.*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+\
        (2-2*nu)*(rb*sinB-y1)*cosB)-a*y3b*cosB*cotB/rb2)*W3+y2*W8/rb/\
        W7*(-cosB/W7**2*(W1*(N1*cosB-a/rb)*cotB+(2-2*nu)*(rb*sinB-y1)*\
        cosB)*W3+cosB/W7*((cosB*y3b/rb+1)*(N1*cosB-a/rb)*cotB+1/2*W1*\
        a/rb**3*2*y3b*cotB+1/2.*(2-2*nu)/rb*sinB*2*y3b*cosB)-a*cosB*\
        cotB/rb2+a*y3b*cosB*cotB/rb2**2*2*y3b))/pi/(1-nu))+\
        b2/2*(1/4*(N1*(((2-2*nu)*cotB**2+nu)*(y3b/rb+1)/W6-((2-2*nu)*cotB**\
        2+1)*cosB*W3/W7)-N1/W6**2.*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/\
        rb+y1**2./W6*W4)*(y3b/rb+1)+N1/W6*(nu-1/2*a*y1*cotB/rb**3*2.*\
        y3b-y1**2./W6**2*W4*(y3b/rb+1)-1/2*y1**2./W6*a/rb**3*2*y3b)+\
        N1*cotB/W7**2.*(z1b*cosB-a*(rb*sinB-y1)/rb/cosB)*W3-N1*cotB/\
        W7*(cosB*sinB-1/2*a/rb2*sinB*2*y3b/cosB+1/2*a*(rb*sinB-y1)/\
        rb**3./cosB*2*y3b)-a/rb**3*y1*cotB+3./2*a*y1*W8*cotB/rb**5*2.*\
        y3b+1./W6*(2*nu+1./rb*(N1*y1*cotB+a)-y1**2./rb/W6*W5-a*y1**2./\
        rb**3)-W8/W6**2.*(2*nu+1./rb*(N1*y1*cotB+a)-y1**2./rb/W6*W5-a*\
        y1**2./rb**3)*(y3b/rb+1)+W8/W6*(-1/2./rb**3.*(N1*y1*cotB+a)*2.*\
        y3b+1/2*y1**2./rb**3./W6*W5*2*y3b+y1**2./rb/W6**2*W5*(y3b/rb+\
        1)+1/2*y1**2./rb2**2./W6*a*2*y3b+3./2*a*y1**2./rb**5*2*y3b)+\
        cotB/W7*(-cosB*sinB+a*y1*y3b/rb**3./cosB+(rb*sinB-y1)/rb*((2-\
        2*nu)*cosB-W1/W7*W9))-W8*cotB/W7**2.*(-cosB*sinB+a*y1*y3b/rb**\
        3./cosB+(rb*sinB-y1)/rb*((2-2*nu)*cosB-W1/W7*W9))*W3+W8*cotB/\
        W7*(a/rb**3./cosB*y1-3./2*a*y1*y3b/rb**5./cosB*2*y3b+1/2./\
        rb2*sinB*2*y3b*((2-2*nu)*cosB-W1/W7*W9)-1/2.*(rb*sinB-y1)/rb**\
        3.*((2-2*nu)*cosB-W1/W7*W9)*2*y3b+(rb*sinB-y1)/rb*(-(cosB*y3b/\
        rb+1)/W7*W9+W1/W7**2*W9*W3+1/2*W1/W7*a/rb**3./cosB*2.*\
        y3b)))/pi/(1-nu))+\
        b3/2*(1/4*(N1*(-y2/W6**2.*(1+a/rb)*(y3b/rb+1)-1/2*y2/W6*a/\
        rb**3*2*y3b+y2*cosB/W7**2*W2*W3+1/2*y2*cosB/W7*a/rb**3*2.*\
        y3b)-y2/rb*(a/rb2+1/W6)+1/2*y2*W8/rb**3.*(a/rb2+1/W6)*2.*\
        y3b-y2*W8/rb*(-a/rb2**2*2*y3b-1/W6**2.*(y3b/rb+1))+y2*cosB/\
        rb/W7*(W1/W7*W2+a*y3b/rb2)-1/2*y2*W8*cosB/rb**3/W7*(W1/\
        W7*W2+a*y3b/rb2)*2*y3b-y2*W8*cosB/rb/W7**2.*(W1/W7*W2+a*\
        y3b/rb2)*W3+y2*W8*cosB/rb/W7*((cosB*y3b/rb+1)/W7*W2-W1/\
        W7**2*W2*W3-1/2*W1/W7*a/rb**3*2*y3b+a/rb2-a*y3b/rb2**2*2.*\
        y3b))/pi/(1-nu))+\
        b1/2*(1/4*((2-2*nu)*(N1*rFib_ry1*cotB-y1/W6**2*W5/rb*y2-y2/W6*\
        a/rb**3*y1+y2*cosB/W7**2*W2*(y1/rb-sinB)+y2*cosB/W7*a/rb**\
        3*y1)-y2*W8/rb**3.*(2*nu/W6+a/rb2)*y1+y2*W8/rb*(-2*nu/W6**\
        2/rb*y1-2*a/rb2**2*y1)-y2*W8*cosB/rb**3/W7*(1-2*nu-W1/W7*\
        W2-a*y3b/rb2)*y1-y2*W8*cosB/rb/W7**2.*(1-2*nu-W1/W7*W2-a*\
        y3b/rb2)*(y1/rb-sinB)+y2*W8*cosB/rb/W7*(-1/rb*cosB*y1/W7*\
        W2+W1/W7**2*W2*(y1/rb-sinB)+W1/W7*a/rb**3*y1+2*a*y3b/rb2**\
        2*y1))/pi/(1-nu))+\
        b2/2*(1/4*((-2+2*nu)*N1*cotB*(1/rb*y1/W6-cosB*(y1/rb-sinB)/W7)-\
        (2-2*nu)/W6*W5+(2-2*nu)*y1**2/W6**2*W5/rb+(2-2*nu)*y1**2/W6*\
        a/rb**3+(2-2*nu)*cosB/W7*W2-(2-2*nu)*z1b/W7**2*W2*(y1/rb-\
        sinB)-(2-2*nu)*z1b/W7*a/rb**3*y1-W8/rb**3.*(N1*cotB-2*nu*y1/\
        W6-a*y1/rb2)*y1+W8/rb*(-2*nu/W6+2*nu*y1**2/W6**2/rb-a/rb2+\
        2*a*y1**2/rb2**2)+W8/W7**2.*(cosB*sinB+W1*cotB/rb*((2-2*nu)*\
        cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*(y1/rb-\
        sinB)-W8/W7*(1/rb2*cosB*y1*cotB*((2-2*nu)*cosB-W1/W7)-W1*\
        cotB/rb**3.*((2-2*nu)*cosB-W1/W7)*y1+W1*cotB/rb*(-1/rb*cosB*\
        y1/W7+W1/W7**2.*(y1/rb-sinB))-a/rb**3.*(sinB-y3b*z1b/rb2-\
        z1b*W1/rb/W7)*y1+a/rb*(-y3b*cosB/rb2+2*y3b*z1b/rb2**2*y1-\
        cosB*W1/rb/W7-z1b/rb2*cosB*y1/W7+z1b*W1/rb**3/W7*y1+z1b*\
        W1/rb/W7**2.*(y1/rb-sinB))))/pi/(1-nu))+\
        b3/2*(1/4*((2-2*nu)*rFib_ry1-(2-2*nu)*y2*sinB/W7**2*W2*(y1/rb-\
        sinB)-(2-2*nu)*y2*sinB/W7*a/rb**3*y1-y2*W8*sinB/rb**3/W7*(1+\
        W1/W7*W2+a*y3b/rb2)*y1-y2*W8*sinB/rb/W7**2.*(1+W1/W7*W2+\
        a*y3b/rb2)*(y1/rb-sinB)+y2*W8*sinB/rb/W7*(1/rb*cosB*y1/\
        W7*W2-W1/W7**2*W2*(y1/rb-sinB)-W1/W7*a/rb**3*y1-2*a*y3b/\
        rb2**2*y1))/pi/(1-nu))
    
    
    
    v23 = b1/2*(1/4*(N1*(((2-2*nu)*cotB**2-nu)*(y3b/rb+1)/W6-((2-2*nu)*\
        cotB**2+1-2*nu)*cosB*W3/W7)+N1/W6**2.*(y1*cotB*(1-W5)+nu*y3b-a+\
        y2**2/W6*W4)*(y3b/rb+1)-N1/W6*(1/2*a*y1*cotB/rb**3*2*y3b+\
        nu-y2**2/W6**2*W4*(y3b/rb+1)-1/2*y2**2/W6*a/rb**3*2*y3b)-N1*\
        sinB*cotB/W7*W2+N1*z1b*cotB/W7**2*W2*W3+1/2*N1*z1b*cotB/W7*\
        a/rb**3*2*y3b-a/rb**3*y1*cotB+3/2*a*y1*W8*cotB/rb**5*2*y3b+\
        1/W6*(-2*nu+1/rb*(N1*y1*cotB-a)+y2**2/rb/W6*W5+a*y2**2./\
        rb**3)-W8/W6**2.*(-2*nu+1/rb*(N1*y1*cotB-a)+y2**2/rb/W6*W5+\
        a*y2**2/rb**3)*(y3b/rb+1)+W8/W6*(-1/2/rb**3.*(N1*y1*cotB-a)*\
        2*y3b-1/2*y2**2/rb**3/W6*W5*2*y3b-y2**2/rb/W6**2*W5*(y3b/\
        rb+1)-1/2*y2**2/rb2**2/W6*a*2*y3b-3/2*a*y2**2/rb**5*2*y3b)+\
        1/W7*(cosB**2-1/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**\
        3-1/rb/W7*(y2**2*cosB**2-a*z1b*cotB/rb*W1))-W8/W7**2.*(cosB**2-\
        1/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/rb**3-1/rb/W7*\
        (y2**2*cosB**2-a*z1b*cotB/rb*W1))*W3+W8/W7*(1/2/rb**3.*(N1*\
        z1b*cotB+a*cosB)*2*y3b-1/rb*N1*sinB*cotB+a*z1b*cotB/rb**3+a*\
        y3b*sinB*cotB/rb**3-3/2*a*y3b*z1b*cotB/rb**5*2*y3b+1/2/rb**\
        3/W7*(y2**2*cosB**2-a*z1b*cotB/rb*W1)*2*y3b+1/rb/W7**2.*(y2**\
        2*cosB**2-a*z1b*cotB/rb*W1)*W3-1/rb/W7*(-a*sinB*cotB/rb*W1+\
        1/2*a*z1b*cotB/rb**3*W1*2*y3b-a*z1b*cotB/rb*(cosB*y3b/rb+\
        1))))/pi/(1-nu))+\
        b2/2*(1/4*((2-2*nu)*N1*rFib_ry3*cotB**2-N1*y2/W6**2.*((W5-1)*cotB+\
        y1/W6*W4)*(y3b/rb+1)+N1*y2/W6*(-1/2*a/rb**3*2*y3b*cotB-y1/\
        W6**2*W4*(y3b/rb+1)-1/2*y1/W6*a/rb**3*2*y3b)+N1*y2*cotB/\
        W7**2*W9*W3+1/2*N1*y2*cotB/W7*a/rb**3/cosB*2*y3b-a/rb**3.*\
        y2*cotB+3/2*a*y2*W8*cotB/rb**5*2*y3b+y2/rb/W6*(N1*cotB-2*\
        nu*y1/W6-a*y1/rb*(1/rb+1/W6))-1/2*y2*W8/rb**3/W6*(N1*\
        cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))*2*y3b-y2*W8/rb/W6**\
        2.*(N1*cotB-2*nu*y1/W6-a*y1/rb*(1/rb+1/W6))*(y3b/rb+1)+y2*\
        W8/rb/W6*(2*nu*y1/W6**2.*(y3b/rb+1)+1/2*a*y1/rb**3.*(1/rb+\
        1/W6)*2*y3b-a*y1/rb*(-1/2/rb**3*2*y3b-1/W6**2.*(y3b/rb+\
        1)))+y2*cotB/rb/W7*((-2+2*nu)*cosB+W1/W7*W9+a*y3b/rb2/cosB)-\
        1/2*y2*W8*cotB/rb**3/W7*((-2+2*nu)*cosB+W1/W7*W9+a*y3b/\
        rb2/cosB)*2*y3b-y2*W8*cotB/rb/W7**2.*((-2+2*nu)*cosB+W1/W7*\
        W9+a*y3b/rb2/cosB)*W3+y2*W8*cotB/rb/W7*((cosB*y3b/rb+1)/\
        W7*W9-W1/W7**2*W9*W3-1/2*W1/W7*a/rb**3/cosB*2*y3b+a/rb2/\
        cosB-a*y3b/rb2**2/cosB*2*y3b))/pi/(1-nu))+\
        b3/2*(1/4*(N1*(-sinB*W3/W7+y1/W6**2.*(1+a/rb)*(y3b/rb+1)+\
        1/2*y1/W6*a/rb**3*2*y3b+sinB/W7*W2-z1b/W7**2*W2*W3-1/2*\
        z1b/W7*a/rb**3*2*y3b)+y1/rb*(a/rb2+1/W6)-1/2*y1*W8/rb**\
        3.*(a/rb2+1/W6)*2*y3b+y1*W8/rb*(-a/rb2**2*2*y3b-1/W6**2.*\
        (y3b/rb+1))-1/W7*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/rb2)-1./\
        rb/W7*(y2**2*cosB*sinB-a*z1b/rb*W1))+W8/W7**2.*(sinB*(cosB-\
        a/rb)+z1b/rb*(1+a*y3b/rb2)-1/rb/W7*(y2**2*cosB*sinB-a*z1b/\
        rb*W1))*W3-W8/W7*(1/2*sinB*a/rb**3*2*y3b+sinB/rb*(1+a*y3b/\
        rb2)-1/2*z1b/rb**3.*(1+a*y3b/rb2)*2*y3b+z1b/rb*(a/rb2-a*\
        y3b/rb2**2*2*y3b)+1/2/rb**3/W7*(y2**2*cosB*sinB-a*z1b/rb*\
        W1)*2*y3b+1/rb/W7**2.*(y2**2*cosB*sinB-a*z1b/rb*W1)*W3-1./\
        rb/W7*(-a*sinB/rb*W1+1/2*a*z1b/rb**3*W1*2*y3b-a*z1b/rb*\
        (cosB*y3b/rb+1))))/pi/(1-nu))+\
        b1/2*(1/4*((2-2*nu)*(N1*rFib_ry2*cotB+1/W6*W5-y2**2/W6**2*W5/\
        rb-y2**2/W6*a/rb**3-cosB/W7*W2+y2**2*cosB/W7**2*W2/rb+y2**2*\
        cosB/W7*a/rb**3)+W8/rb*(2*nu/W6+a/rb2)-y2**2*W8/rb**3.*(2*\
        nu/W6+a/rb2)+y2*W8/rb*(-2*nu/W6**2/rb*y2-2*a/rb2**2*y2)+\
        W8*cosB/rb/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-y2**2*W8*cosB/\
        rb**3/W7*(1-2*nu-W1/W7*W2-a*y3b/rb2)-y2**2*W8*cosB/rb2/W7**\
        2.*(1-2*nu-W1/W7*W2-a*y3b/rb2)+y2*W8*cosB/rb/W7*(-1/rb*\
        cosB*y2/W7*W2+W1/W7**2*W2/rb*y2+W1/W7*a/rb**3*y2+2*a*\
        y3b/rb2**2*y2))/pi/(1-nu))+\
        b2/2*(1/4*((-2+2*nu)*N1*cotB*(1/rb*y2/W6-cosB/rb*y2/W7)+(2-\
        2*nu)*y1/W6**2*W5/rb*y2+(2-2*nu)*y1/W6*a/rb**3*y2-(2-2*\
        nu)*z1b/W7**2*W2/rb*y2-(2-2*nu)*z1b/W7*a/rb**3*y2-W8/rb**\
        3.*(N1*cotB-2*nu*y1/W6-a*y1/rb2)*y2+W8/rb*(2*nu*y1/W6**2./\
        rb*y2+2*a*y1/rb2**2*y2)+W8/W7**2.*(cosB*sinB+W1*cotB/rb*((2-\
        2*nu)*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))/\
        rb*y2-W8/W7*(1/rb2*cosB*y2*cotB*((2-2*nu)*cosB-W1/W7)-W1*\
        cotB/rb**3.*((2-2*nu)*cosB-W1/W7)*y2+W1*cotB/rb*(-cosB/rb*\
        y2/W7+W1/W7**2/rb*y2)-a/rb**3.*(sinB-y3b*z1b/rb2-z1b*W1/\
        rb/W7)*y2+a/rb*(2*y3b*z1b/rb2**2*y2-z1b/rb2*cosB*y2/W7+\
        z1b*W1/rb**3/W7*y2+z1b*W1/rb2/W7**2*y2)))/pi/(1-nu))+\
        b3/2*(1/4*((2-2*nu)*rFib_ry2+(2-2*nu)*sinB/W7*W2-(2-2*nu)*y2**2*\
        sinB/W7**2*W2/rb-(2-2*nu)*y2**2*sinB/W7*a/rb**3+W8*sinB/rb/\
        W7*(1+W1/W7*W2+a*y3b/rb2)-y2**2*W8*sinB/rb**3/W7*(1+W1/\
        W7*W2+a*y3b/rb2)-y2**2*W8*sinB/rb2/W7**2.*(1+W1/W7*W2+a*\
        y3b/rb2)+y2*W8*sinB/rb/W7*(1/rb*cosB*y2/W7*W2-W1/W7**2.*\
        W2/rb*y2-W1/W7*a/rb**3*y2-2*a*y3b/rb2**2*y2))/pi/(1-nu))
    
    #print('v23_',v23)
    return v11,v22,v33,v12,v13,v23
        
    
def TensTrans(Txx1,Tyy1,Tzz1,Txy1,Txz1,Tyz1,A):
    Txx2 = A[0][0]*A[0][0]*Txx1+2*A[0][0]*A[1][0]*Txy1+2*A[0][0]*A[2][0]*Txz1+2*A[1][0]*A[2][0]*Tyz1+\
        A[1][0]*A[1][0]*Tyy1+A[2][0]*A[2][0]*Tzz1;
    
    
    Tyy2 = A[0][1]*A[0][1]*Txx1+2*A[0][1]*A[1][1]*Txy1+2*A[0][1]*A[2][1]*Txz1+2*A[1][1]*A[2][1]*Tyz1+\
        A[1][1]*A[1][1]*Tyy1+A[2][1]*A[2][1]*Tzz1;
    Tzz2 = A[0][2]*A[0][2]*Txx1+2*A[0][2]*A[1][2]*Txy1+2*A[0][2]*A[2][2]*Txz1+2*A[1][2]*A[2][2]*Tyz1+\
        A[1][2]*A[1][2]*Tyy1+A[2][2]*A[2][2]*Tzz1;
    Txy2 = A[0][0]*A[0][1]*Txx1+(A[0][0]*A[1][1]+A[0][1]*A[1][0])*Txy1+(A[0][0]*A[2][1]+
        A[0][1]*A[2][0])*Txz1+(A[2][1]*A[1][0]+A[2][0]*A[1][1])*Tyz1+A[1][1]*A[1][0]*Tyy1+\
        A[2][0]*A[2][1]*Tzz1
    Txz2 = A[0][0]*A[0][2]*Txx1+(A[0][0]*A[1][2]+A[0][2]*A[1][0])*Txy1+(A[0][0]*A[2][2]+
        A[0][2]*A[2][0])*Txz1+(A[2][2]*A[1][0]+A[2][0]*A[1][2])*Tyz1+A[1][2]*A[1][0]*Tyy1+\
        A[2][0]*A[2][2]*Tzz1
    Tyz2 = A[0][1]*A[0][2]*Txx1+(A[0][2]*A[1][1]+A[0][1]*A[1][2])*Txy1+(A[0][2]*A[2][1]+
        A[0][1]*A[2][2])*Txz1+(A[2][1]*A[1][2]+A[2][2]*A[1][1])*Tyz1+A[1][1]*A[1][2]*Tyy1+\
        A[2][1]*A[2][2]*Tzz1
    return Txx2,Tyy2,Tzz2,Txy2,Txz2,Tyz2    
    

        
    
    
    


def AngSetupFSC_S(X,Y,Z,B_vec,PA,PB,mu,lambda_):
    #AngSetupFSC_S calculates the Free Surface Correction to strains

    nu = 1/(1+lambda_/mu)/2; #Poisson's ratio

    #Calculate TD side vector and the angle of the angular dislocation pair
    SideVec = PB-PA
    eZ = np.array([0,0,1])
    beta = acos(-np.dot(SideVec,eZ)/norm(SideVec))
    #eps=2.2204e-16
    eps=1e-4
    if(abs(beta)<eps or abs(pi-beta)<eps):
        Stress = np.zeros((6,len(X)))
        Strain = np.zeros((6,len(X)))
    else:
        ey1 = [SideVec[0],SideVec[1],0]
        ey1 = ey1/norm(ey1)
        ey3 = -eZ
        ey2 = np.cross(ey3,ey1)
        A = np.array([ey1,ey2,ey3]); # Transformation matrix
        
        # Transform coordinates from EFCS to the first ADCS
        yA=np.dot(np.array([X-PA[0],Y-PA[1],Z-PA[2]]).transpose(),A)
        yAB=np.dot(SideVec,A)
        yB = yA-yAB
        bv=np.dot(A,B_vec)
        #print(bv)
        
        #Determine the best arteact-free configuration for the calculation
        #points near the free furface
        I = (beta*yA[:,0])>=0
        NI = ~I
        #print(I,beta)
        #For singularities at surface
        v11A = np.zeros(len(X))
        v22A = np.zeros(len(X))
        v33A = np.zeros(len(X))
        v12A = np.zeros(len(X))
        v13A = np.zeros(len(X))
        v23A = np.zeros(len(X))

        v11B = np.zeros(len(X))
        v22B = np.zeros(len(X))
        v33B = np.zeros(len(X))
        v12B = np.zeros(len(X))
        v13B = np.zeros(len(X))
        v23B = np.zeros(len(X))
        
        # Configuration I
        v11A[I],v22A[I],v33A[I],v12A[I],v13A[I],v23A[I]=AngDisStrainFSC(-yA[I,0],-yA[I,1],yA[I,2],\
                                                pi-beta,-bv[0],-bv[1],bv[2],nu,-PA[2])
        v13A[I] = -v13A[I]
        v23A[I] = -v23A[I]
        #print(v11A)
        
        v11B[I],v22B[I],v33B[I],v12B[I],v13B[I],v23B[I]=AngDisStrainFSC(-yB[I,0],-yB[I,1],yB[I,2],\
                                                pi-beta,-bv[0],-bv[1],bv[2],nu,-PB[2])
        v13B[I] = -v13B[I]
        v23B[I] = -v23B[I]
        
        if(np.sum(NI)>0):
            #print('PB',PB)
            v11A[NI],v22A[NI],v33A[NI],v12A[NI],v13A[NI],v23A[NI] = \
                AngDisStrainFSC(yA[NI,0],yA[NI,1],yA[NI,2],beta,bv[0],bv[1],bv[2],nu,-PA[2])
                

            v11B[NI],v22B[NI],v33B[NI],v12B[NI],v13B[NI],v23B[NI] = \
                AngDisStrainFSC(yB[NI,0],yB[NI,1],yB[NI,2],beta,bv[0],bv[1],bv[2],nu,-PB[2])
                
        
        # Calculate total Free Surface Correction to strains in ADCS
        v11 = v11B-v11A
        v22 = v22B-v22A
        v33 = v33B-v33A
        v12 = v12B-v12A
        v13 = v13B-v13A
        v23 = v23B-v23A
        #print(v11,v22,v33,v12,v13,v23)
        Exx,Eyy,Ezz,Exy,Exz,Eyz = TensTrans(v11,v22,v33,v12,v13,v23,A)
        #print(Exx,Eyy,Ezz,Exy,Exz,Eyz)
        # Calculate total Free Surface Correction to stresses in EFCS
        Sxx = 2*mu*Exx+lambda_*(Exx+Eyy+Ezz)
        Syy = 2*mu*Eyy+lambda_*(Exx+Eyy+Ezz)
        Szz = 2*mu*Ezz+lambda_*(Exx+Eyy+Ezz)
        Sxy = 2*mu*Exy
        Sxz = 2*mu*Exz
        Syz = 2*mu*Eyz

        Strain = np.array([Exx,Eyy,Ezz,Exy,Exz,Eyz])
        Stress = np.array([Sxx,Syy,Szz,Sxy,Sxz,Syz])
        
    return Stress,Strain

    

def TDstress_HarFunc(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda_):
    bx = Ts
    by = Ss
    bz = Ds 
    X=X.ravel()
    Y=Y.ravel()
    Z=Z.ravel()
    
    Vnorm = np.cross(P2-P1,P3-P1)
    Vnorm = Vnorm/norm(Vnorm)

    eX,eY,eZ = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
    Vstrike = np.cross(eZ,Vnorm)

    if(norm(Vstrike)==0):
        Vstrike = eX*Vnorm[2]

    Vstrike = Vstrike/norm(Vstrike)
    Vdip = np.cross(Vnorm,Vstrike)
    
    A = np.array([Vnorm,Vstrike,Vdip]).transpose()
    B_vec=np.dot(A,np.array([bx,by,bz]))
    #print(B_vec)
    Stress1,Strain1 = AngSetupFSC_S(X,Y,Z,B_vec,P1,P2,mu,lambda_) # P1P2
    Stress2,Strain2 = AngSetupFSC_S(X,Y,Z,B_vec,P2,P3,mu,lambda_) # P2P3
    Stress3,Strain3 = AngSetupFSC_S(X,Y,Z,B_vec,P3,P1,mu,lambda_) # P3P1
    #Calculate total harmonic function contribution to strains and stresses
    Stress = Stress1+Stress2+Stress3
    Strain = Strain1+Strain2+Strain3
    #print(Stress)
    return Stress,Strain
    
def TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu,lambda_):

    if(np.max(Z)>0 or P1[2]>0 or P2[2]>0 or P3[2]>0):
        print('Half-space solution: Z coordinates must be negative!')

    #Calculate main dislocation contribution to displacements
    StsMS,StrMS=TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu,lambda_)
    

    StsFSC,StrFSC=TDstress_HarFunc(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu,lambda_)

    #Calculate image dislocation contribution to displacements
    P1[2] = -P1[2]
    P2[2] = -P2[2]
    P3[2] = -P3[2]
    StsIS,StrIS = TDstressFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu,lambda_)

    if(P1[2]==0 and P2[2]==0 and P3[2]==0):
        StsIS[:,4] = -StsIS[:,4]
        StsIS[:,5] = -StsIS[:,5]
        StrIS[:,4] = -StrIS[:,4]
        StrIS[:,5] = -StrIS[:,5]

    # Calculate the complete displacement vector components in EFCS
    Stress = StsMS+StsIS+StsFSC
    Strain = StrMS+StrIS+StrFSC

    return Stress,Strain


#def get_Sprojection()

#def getmatrixA()

