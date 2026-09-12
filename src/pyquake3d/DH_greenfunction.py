import numpy as np
from numpy.linalg import norm
from math import *
import matplotlib.pyplot as plt

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

def TDSetupD(coord,alpha,bx,by,bz,nu,TriVertex,SideVec):
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
    
    u,v0,w0=AngDisDisp(coord1,-np.pi+alpha,bx,by1,bz1,nu)
    #print(u,v,w)
    # Transform displacements from ADCS into TDCS
    r3 = np.array([v0,w0]).transpose()
    r3=np.dot(r3,A)
    v = r3[:,0]
    w = r3[:,1]
    #print(u,v,w)
    return u,v,w
    


    
def AngDisDisp(coord,alpha,bx,by,bz,nu):
    cosA = cos(alpha)
    sinA = sin(alpha)
    x=coord[:,0]
    y=coord[:,1]
    z=coord[:,2]
    eta = y*cosA-z*sinA
    zeta = y*sinA+z*cosA  
    r = np.linalg.norm(coord,axis=1);
    
    # Avoid complex results for the logarithmic terms
    zeta[zeta>r] = r[zeta>r];
    z[z>r] = r[z>r];
    
#     for i in range(len(r)):
#         print(eta[i],zeta[i],r[i])
    
    ux = bx/8/pi/(1-nu)*(x*y/r/(r-z)-x*eta/r/(r-zeta))
    #vx = bx/8/pi/(1-nu)*(eta*sinA/(r-zeta)-y*eta/r/(r-zeta)+y*y/r/(r-z))
    vx = bx/8/pi/(1-nu)*(eta*sinA/(r-zeta)-y*eta/r/(r-zeta)+y*y/r/(r-z)+(1-2*nu)*(cosA*np.log(r-zeta)-np.log(r-z)))
    wx = bx/8/pi/(1-nu)*(eta*cosA/(r-zeta)-y/r-eta*z/r/(r-zeta)-(1-2*nu)*sinA*np.log(r-zeta));
    uy = by/8/pi/(1-nu)*(x*x*cosA/r/(r-zeta)-x*x/r/(r-z)-(1-2*nu)*(cosA*np.log(r-zeta)-np.log(r-z)));
    vy = by*x/8/pi/(1-nu)*(y*cosA/r/(r-zeta)-sinA*cosA/(r-zeta)-y/r/(r-z));
    wy = by*x/8/pi/(1-nu)*(z*cosA/r/(r-zeta)-cosA*cosA/(r-zeta)+1/r);
    
    uz = bz*sinA/8/pi/(1-nu)*((1-2*nu)*np.log(r-zeta)-x*x/r/(r-zeta));
    vz = bz*x*sinA/8/pi/(1-nu)*(sinA/(r-zeta)-y/r/(r-zeta));
    wz = bz*x*sinA/8/pi/(1-nu)*(cosA/(r-zeta)-z/r/(r-zeta));
    
    u = ux+uy+uz
    v = vx+vy+vz
    w = wx+wy+wz
    return u,v,w
    
def TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu):
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
    e12 = (p2-p1)/norm(p2-p1);
    e13 = (p3-p1)/norm(p3-p1);
    e23 = (p3-p2)/norm(p3-p2);


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
    
    

    #Xp
    if(np.count_nonzero(casepLog)>0):
        #Calculate first angular dislocation contribution
        u1Tp,v1Tp,w1Tp=TDSetupD(Xp,A,bx,by,bz,nu,p1,-e13)
        
        #Calculate second angular dislocation contribution
        u2Tp,v2Tp,w2Tp=TDSetupD(Xp,B,bx,by,bz,nu,p2,e12)
        
        
        #Calculate third angular dislocation contribution
        u3Tp,v3Tp,w3Tp=TDSetupD(Xp,C,bx,by,bz,nu,p3,e23)

    if(np.count_nonzero(casenLog)>0): 
        #Calculate first angular dislocation contribution
        u1Tn,v1Tn,w1Tn=TDSetupD(Xn,A,bx,by,bz,nu,p1,e13)
        
        
        #Calculate second angular dislocation contribution
        u2Tn,v2Tn,w2Tn=TDSetupD(Xn,B,bx,by,bz,nu,p2,-e12)
        #Calculate third angular dislocation contribution
        u3Tn,v3Tn,w3Tn=TDSetupD(Xn,C,bx,by,bz,nu,p3,-e23)
        #print(u3Tn,v3Tn,w3Tn)

    u=np.zeros(len(Trimode))
    v=np.zeros(len(Trimode))
    w=np.zeros(len(Trimode))
    if(np.count_nonzero(casepLog)>0):
        u[casepLog]=u1Tp+u2Tp+u3Tp
        v[casepLog]=v1Tp+v2Tp+v3Tp
        w[casepLog]=w1Tp+w2Tp+w3Tp
    if(np.count_nonzero(casenLog)>0): 
        u[casenLog]=u1Tn+u2Tn+u3Tn
        v[casenLog]=v1Tn+v2Tn+v3Tn
        w[casenLog]=w1Tn+w2Tn+w3Tn
    if(np.count_nonzero(casezLog)>0): 
        u[casezLog]=np.nan
        v[casezLog]=np.nan
        w[casezLog]=np.nan

    
    
    a = p1-X1
    b = p2-X1
    c = p3-X1

    na = norm(a,axis=1)
    nb = norm(b,axis=1)
    nc = norm(c,axis=1)

    Fi = -2 * np.arctan2(np.sum(a*np.cross(b, c),axis=1),(na * nb * nc + np.sum(a * b,axis=1) * nc + np.sum(a * c,axis=1) * nb + np.sum(b * c,axis=1) * na)) / (4 * np.pi)
    

    #print(v)
    #Calculate the complete displacement vector components in TDCS
    u = bx*Fi+u
    v = by*Fi+v
    w = bz*Fi+w
    
    # Transform the complete displacement vector components from TDCS into EFCS
    At = np.array([Vnorm,Vstrike,Vdip])
    X_EFCS=np.dot(np.array([u,v,w]).transpose(),At)
    
    return X_EFCS[:,0],X_EFCS[:,1],X_EFCS[:,2]



def TDdisp_HarFunc(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu):
    bx = Ts
    by = Ss
    bz = Ds 
    X=X.ravel()
    Y=Y.ravel()
    Z=Z.ravel()
    
    Vnorm = np.cross(P2-P1,P3-P1)
    Vnorm = Vnorm/norm(Vnorm)

    #eY,eZ = np.array([0,1,0]),np.array([0,0,1])
    eX,eY,eZ = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
    Vstrike = np.cross(eZ,Vnorm)

    if(norm(Vstrike)==0):
        Vstrike = eX*Vnorm[2]

    Vstrike = Vstrike/norm(Vstrike)
    Vdip = np.cross(Vnorm,Vstrike)
    
    At = np.array([Vnorm,Vstrike,Vdip]).transpose()
    B_vec=np.dot(At,np.array([bx,by,bz]))
    
    
    u_vec1=AngSetupFSC(X,Y,Z,B_vec,P1,P2,nu)
    u_vec2=AngSetupFSC(X,Y,Z,B_vec,P2,P3,nu)
    u_vec3=AngSetupFSC(X,Y,Z,B_vec,P3,P1,nu)
    
    # Calculate total harmonic function contribution to displacements
    ue = u_vec1[:,0]+u_vec2[:,0]+u_vec3[:,0]
    un = u_vec1[:,1]+u_vec2[:,1]+u_vec3[:,1]
    uv = u_vec1[:,2]+u_vec2[:,2]+u_vec3[:,2]
    return ue,un,uv
    
    
    

    
def AngSetupFSC(X,Y,Z,B_vec,PA,PB,nu):
    SideVec = PB-PA
    eZ = np.array([0,0,1])
    beta = np.arccos(np.dot(-SideVec, eZ)/norm(SideVec));
    eps=2.2e-10
    if(abs(beta)<eps or abs(pi-beta)<eps):
        ue=np.zeros(len(X))
        un=np.zeros(len(X))
        uv=np.zeros(len(X))
    else:
        ey1 = np.zeros(3)
        ey1[:2]=SideVec[:2]
        ey1 = ey1/norm(ey1)
        ey3 = -eZ
        ey2 = np.cross(ey3,ey1)
        At = np.array([ey1,ey2,ey3])
        
        yA_vec=np.dot(np.array([X-PA[0],Y-PA[1],Z-PA[2]]).transpose(),At)
        yAB_vec=np.dot(SideVec.transpose(),At)
        yB_vec=yA_vec-yAB_vec
        
        b_vec=np.dot(B_vec.transpose(),At)
        
        # Determine the best arteact-free configuration for the calculation
        # points near the free furface
        I = np.dot(beta,yA_vec[:,0])>=0
        negI = np.logical_not(I)
        #print(I)
        
        v1A,v2A,v3A=np.zeros(len(I)),np.zeros(len(I)),np.zeros(len(I))
        v1B,v2B,v3B=np.zeros(len(I)),np.zeros(len(I)),np.zeros(len(I))
        
        # Configuration I
        v1A[I],v2A[I],v3A[I]=AngDisDispFSC(yA_vec[I],-pi+beta,b_vec,nu,-PA[2]);
        v1B[I],v2B[I],v3B[I]=AngDisDispFSC(yB_vec[I],-pi+beta,b_vec,nu,-PB[2]);
        
        # Configuration II
        if(np.any(negI)==True):
            #print('!!!!!!')
            v1A[negI],v2A[negI],v3A[negI]=AngDisDispFSC(yA_vec[negI],beta,b_vec,nu,-PA[2]);
            v1B[negI],v2B[negI],v3B[negI]=AngDisDispFSC(yB_vec[negI],beta,b_vec,nu,-PB[2]);
        
        #print(v1B[I],v2B[I],v3B[I])
        v1 = v1B-v1A
        v2 = v2B-v2A
        v3 = v3B-v3A
        #print(v1,v2,v3)
        u_vec=np.dot(np.array([v1,v2,v3]).transpose(),At.transpose())
        return u_vec
        
    
    
def AngDisDispFSC(y_vec,beta,b_vec,nu,a):
    sinB = sin(beta)
    cosB = cos(beta)
    cotB = 1.0 / tan(beta)
    y1=y_vec[:,0]
    y2=y_vec[:,1]
    y3=y_vec[:,2]
    b1=b_vec[0]
    b2=b_vec[1]
    b3=b_vec[2]
    y3b = y3+2*a
    z1b = y1*cosB+y3b*sinB
    z3b = -y1*sinB+y3b*cosB
    r2b = y1*y1+y2*y2+y3b*y3b
    rb = np.sqrt(r2b)
    r3b=rb*rb*rb
    Fib = 2*np.arctan(-y2/(-(rb+y3b)/tan(beta/2.0)+y1)); #The Burgers' function
    
    
    v1cb1 = b1/4/pi/(1-nu)*(-2*(1-nu)*(1-2*nu)*Fib*cotB*cotB+(1-2*nu)*y2/
            (rb+y3b)*((1-2*nu-a/rb)*cotB-y1/(rb+y3b)*(nu+a/rb))+(1-2*nu)*
            y2*cosB*cotB/(rb+z3b)*(cosB+a/rb)+a*y2*(y3b-a)*cotB/r3b+y2*
            (y3b-a)/(rb*(rb+y3b))*(-(1-2*nu)*cotB+y1/(rb+y3b)*(2*nu+a/rb)+
            a*y1/r2b)+y2*(y3b-a)/(rb*(rb+z3b))*(cosB/(rb+z3b)*((rb*
            cosB+y3b)*((1-2*nu)*cosB-a/rb)*cotB+2*(1-nu)*(rb*sinB-y1)*cosB)-
            a*y3b*cosB*cotB/r2b))
    
    
    v2cb1 = b1/4/pi/(1-nu)*((1-2*nu)*((2*(1-nu)*cotB*cotB-nu)*np.log(rb+y3b)-(2*
            (1-nu)*cotB*cotB+1-2*nu)*cosB*np.log(rb+z3b))-(1-2*nu)/(rb+y3b)*(y1*
            cotB*(1-2*nu-a/rb)+nu*y3b-a+y2*y2/(rb+y3b)*(nu+a/rb))-(1-2*
            nu)*z1b*cotB/(rb+z3b)*(cosB+a/rb)-a*y1*(y3b-a)*cotB/r3b+
            (y3b-a)/(rb+y3b)*(-2*nu+1./rb*((1-2*nu)*y1*cotB-a)+y2*y2/(rb*
            (rb+y3b))*(2*nu+a/rb)+a*y2*y2/r3b)+(y3b-a)/(rb+z3b)*(cosB*cosB-
            1./rb*((1-2*nu)*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/r3b-1./(rb*
            (rb+z3b))*(y2*y2*cosB*cosB-a*z1b*cotB/rb*(rb*cosB+y3b))));
    
    v3cb1 = b1/4/pi/(1-nu)*(2*(1-nu)*(((1-2*nu)*Fib*cotB)+(y2/(rb+y3b)*(2*
            nu+a/rb))-(y2*cosB/(rb+z3b)*(cosB+a/rb)))+y2*(y3b-a)/rb*(2*
            nu/(rb+y3b)+a/(rb*rb))+y2*(y3b-a)*cosB/(rb*(rb+z3b))*(1-2*nu-
            (rb*cosB+y3b)/(rb+z3b)*(cosB+a/rb)-a*y3b/rb/rb));
    
    
    v1cb2 = b2/4/pi/(1-nu)*((1-2*nu)*((2*(1-nu)*cotB*cotB+nu)*np.log(rb+y3b)-(2*
            (1-nu)*cotB*cotB+1)*cosB*np.log(rb+z3b))+(1-2*nu)/(rb+y3b)*(-(1-2*nu)*
            y1*cotB+nu*y3b-a+a*y1*cotB/rb+y1*y1/(rb+y3b)*(nu+a/rb))-(1-2*
            nu)*cotB/(rb+z3b)*(z1b*cosB-a*(rb*sinB-y1)/(rb*cosB))-a*y1*
            (y3b-a)*cotB/r3b+(y3b-a)/(rb+y3b)*(2*nu+1./rb*((1-2*nu)*y1*
            cotB+a)-y1*y1/(rb*(rb+y3b))*(2*nu+a/rb)-a*y1*y1/r3b)+(y3b-a)*
            cotB/(rb+z3b)*(-cosB*sinB+a*y1*y3b/(r3b*cosB)+(rb*sinB-y1)/
            rb*(2*(1-nu)*cosB-(rb*cosB+y3b)/(rb+z3b)*(1+a/(rb*cosB)))));
    
    
    v2cb2 = b2/4/pi/(1-nu)*(2*(1-nu)*(1-2*nu)*Fib*cotB*cotB+(1-2*nu)*y2/
            (rb+y3b)*(-(1-2*nu-a/rb)*cotB+y1/(rb+y3b)*(nu+a/rb))-(1-2*nu)*
            y2*cotB/(rb+z3b)*(1+a/(rb*cosB))-a*y2*(y3b-a)*cotB/r3b+y2*
            (y3b-a)/(rb*(rb+y3b))*((1-2*nu)*cotB-2*nu*y1/(rb+y3b)-a*y1/rb*
            (1./rb+1/(rb+y3b)))+y2*(y3b-a)*cotB/(rb*(rb+z3b))*(-2*(1-nu)*
            cosB+(rb*cosB+y3b)/(rb+z3b)*(1+a/(rb*cosB))+a*y3b/(r2b*cosB)))
    
    v3cb2 = b2/4/pi/(1-nu)*(-2*(1-nu)*(1-2*nu)*cotB*(np.log(rb+y3b)-cosB*
            np.log(rb+z3b))-2*(1-nu)*y1/(rb+y3b)*(2*nu+a/rb)+2*(1-nu)*z1b/(rb+
            z3b)*(cosB+a/rb)+(y3b-a)/rb*((1-2*nu)*cotB-2*nu*y1/(rb+y3b)-a*
            y1/r2b)-(y3b-a)/(rb+z3b)*(cosB*sinB+(rb*cosB+y3b)*cotB/rb*
            (2*(1-nu)*cosB-(rb*cosB+y3b)/(rb+z3b))+a/rb*(sinB-y3b*z1b/
            r2b-z1b*(rb*cosB+y3b)/(rb*(rb+z3b)))))
    
    
    v1cb3 = b3/4/pi/(1-nu)*((1-2*nu)*(y2/(rb+y3b)*(1+a/rb)-y2*cosB/(rb+
            z3b)*(cosB+a/rb))-y2*(y3b-a)/rb*(a/r2b+1./(rb+y3b))+y2*
            (y3b-a)*cosB/(rb*(rb+z3b))*((rb*cosB+y3b)/(rb+z3b)*(cosB+a/
            rb)+a*y3b/r2b));
    
    
    v2cb3 = b3/4/pi/(1-nu)*((1-2*nu)*(-sinB*np.log(rb+z3b)-y1/(rb+y3b)*(1+a/
            rb)+z1b/(rb+z3b)*(cosB+a/rb))+y1*(y3b-a)/rb*(a/r2b+1./(rb+
            y3b))-(y3b-a)/(rb+z3b)*(sinB*(cosB-a/rb)+z1b/rb*(1+a*y3b/
            r2b)-1./(rb*(rb+z3b))*(y2*y2*cosB*sinB-a*z1b/rb*(rb*cosB+y3b))))
    
    v3cb3 = b3/4/pi/(1-nu)*(2*(1-nu)*Fib+2*(1-nu)*(y2*sinB/(rb+z3b)*(cosB+
            a/rb))+y2*(y3b-a)*sinB/(rb*(rb+z3b))*(1+(rb*cosB+y3b)/(rb+
            z3b)*(cosB+a/rb)+a*y3b/r2b))
    
    v1 = v1cb1+v1cb2+v1cb3
    v2 = v2cb1+v2cb2+v2cb3
    v3 = v3cb1+v3cb2+v3cb3
    
    return v1,v2,v3
            
def TDdispHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu):

    if(np.max(Z)>0 or P1[2]>0 or P2[2]>0 or P3[2]>0):
        print('Half-space solution: Z coordinates must be negative!')

    

    #Calculate main dislocation contribution to displacements
    ueMS,unMS,uvMS=TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)



    ueFSC,unFSC,uvFSC=TDdisp_HarFunc(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)

    #Calculate image dislocation contribution to displacements
    P1[2] = -P1[2]
    P2[2] = -P2[2]
    P3[2] = -P3[2]
    ueIS,unIS,uvIS = TDdispFS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,nu)

    if(P1[2]==0 and P2[2]==0 and P3[2]==0):
        uvIS = -uvIS

    # Calculate the complete displacement vector components in EFCS
    ue = ueMS+ueIS+ueFSC
    un = unMS+unIS+unFSC
    uv = uvMS+uvIS+uvFSC

    if(P1[2]==0 and P2[2]==0 and P3[2]==0):
        ue = -ue
        un = -un
        uv = -uv
    
    return ue,un,uv

    
    


    
    

    