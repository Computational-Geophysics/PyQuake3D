"""
Quasi-Dynamic simulation with viscosity effects module.

This module extends the base QDsim class to include viscosity modeling
for earthquake rupture propagation simulations.
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
import sys
import time
import ctypes
import logging
from datetime import datetime
from typing import Tuple, Optional
from collections import deque

import scipy.interpolate as interpolate
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import lu_factor, lu_solve

import pyvista as pv
import vtk
import joblib
import psutil

from concurrent.futures import ProcessPoolExecutor, as_completed
from mpi4py import MPI

import pyquake3d.readmsh as readmsh
import pyquake3d.Hmatrix as Hmat
import pyquake3d.QDsim as QDsim
from pyquake3d.config import comm, rank, size

import greenfunctions_lib.laplace_lib as laplace_lib
import greenfunctions_lib.kelvin_stress_lib as kelvin_stress_lib
import greenfunctions_lib.kelvin_stress_potential_lib as kelvin_stress_potential_lib


#if(useC==True):
'''Read C++ compiled dynamic library to calculate green's functions'''
try: 
    try:
        from importlib.resources import files
    except ImportError:
        # Python 3.8 Or earlier
        from importlib_resources import files
    _so_path = files("pyquake3d") / "TDstressFS_C.so" 
    lib = ctypes.CDLL(str(_so_path))
    #lib = ctypes.CDLL('src/TDstressFS_C.so')
    #lib = ctypes.CDLL('src/Dll1.dll')
    # Define the interface parameter types.
    lib.TDstressFS_C.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # X
        ctypes.POINTER(ctypes.c_double),  # Y
        ctypes.POINTER(ctypes.c_double),  # Z
        ctypes.c_size_t,                  # n
        ctypes.POINTER(ctypes.c_double),  # P1
        ctypes.POINTER(ctypes.c_double),  # P2
        ctypes.POINTER(ctypes.c_double),  # P3
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # Ss, Ds, Ts
        ctypes.c_double, ctypes.c_double,                  # mu, lambda_
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_double),  # stress
        ctypes.POINTER(ctypes.c_double),  # strain
    ]

    lib.TDstressEachSourceAtReceiver_C.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # x, y, z
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),  # P1, P2, P3
        ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,  # Ss, Ds, Ts
        ctypes.c_double, ctypes.c_double,  # mu, lambda
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_double),  # stress_out
        ctypes.POINTER(ctypes.c_double)   # strain_out
    ]
except AttributeError as e:
    print(f"Failed to load C dynamic libaray")





from typing import Tuple, Optional
def triangle_dislocation_integral_stress_Kfv_python(
    receivers: np.ndarray,              # shape (n, 3) 或 (3, n) 都可以，會自動轉換
    p1: np.ndarray,                      # 3-element array，三角形第一點 [x,y,z]
    p2: np.ndarray,                      # 3-element array
    p3: np.ndarray,                      # 3-element array
    ss: float = 0.0,                     # strike-slip 分量
    ds: float = 0.0,                     # dip-slip 分量
    ts: float = 0.0,                     # tensile/opening 分量
    mu: float = 0.3,                     # 剪切模量 (Pa)
    lambda_: float = 0.3,                # 拉梅常數 λ (Pa)
    halfspace: bool = False              # 是否使用 half-space 模型
) -> Tuple[np.ndarray, np.ndarray]:
    # 載入共享庫（只載入一次）

    

    # 確保輸入是 numpy 陣列，且資料類型正確
    receivers = np.asarray(receivers, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    if receivers.ndim != 2 or receivers.shape[1] not in (3,):
        raise ValueError("receivers 應為 shape (n,3) 的二維陣列")

    n = receivers.shape[0]

    # 統一轉成 (n,3) 格式
    if receivers.shape[1] != 3:
        receivers = receivers.T.copy()
    X,Y,Z=receivers[:, 0], receivers[:, 1], receivers[:, 2]
        
    n=len(X)
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    Z = np.array(Z, dtype=np.float64)

    X = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    Y = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    Z = Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    P1 = p1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    P2 = p2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    P3 = p3.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


    stress = np.zeros(n * 6)
    strain = np.zeros(n * 6)

    # 呼叫 C 函數
    lib.TDstressFS_C(
        X, Y, Z,
        ctypes.c_size_t(n),
        P1, P2, P3,
        ctypes.c_double(ss),
        ctypes.c_double(ds),
        ctypes.c_double(ts),
        ctypes.c_double(mu),
        ctypes.c_double(lambda_),
        ctypes.c_bool(halfspace),
        stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        strain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    stress=stress.reshape([n,6])
    strain=strain.reshape([n,6])
    return stress, strain





class QDsimvicosity(QDsim.QDsim):
    def __init__(self,elelstV,nodelst,elelstF,Para):
        super().__init__(elelstF,nodelst,Para)
        self.Para0=Para
        #parameter define
        jud_ele_order=self.Para0['Node_order']
        self.vicosity=self.Para0['viscosity']
        #self.vicosity2=self.Para0['viscosity2']
        self.elelstV=elelstV
        eleVecV,xgV=readmsh.get_eleVec(nodelst,elelstV,jud_ele_order)
        self.eleVecV=eleVecV
        self.eleVecF=self.eleVec
        self.elelstF=elelstF
        self.xgF=self.xg
        self.xgV=np.array(xgV, dtype=np.float64)
        self.VFdot1=np.zeros(len(self.eleVecF))
        self.VFdot2=np.zeros(len(self.eleVecF))
        self.VFdot3=np.zeros(len(self.eleVecF))
        self.vicosity0=1e21
        #print(np.min(self.xgV[:,2]),np.min(self.xgF[:,2]))
        
        #self.calc_visco_M()
        # self.calc_disloc_M()

    def init_vicosity_stress(self):
        N=len(self.eleVecV)
        self.EBforce=np.zeros([N,3])
        self.Tra_vicos=np.zeros([N,3])
        trac_nor=self.Para0['Vertical principal stress value']

        def triangle_area(A, B, C):
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross)
            return area
        
        # sigmaxx=np.dot(self.Kfv1[:,:,0],self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,0],self.slipv2[self.local_index])
        # sigmayy=np.dot(self.Kfv1[:,:,1],self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,1],self.slipv2[self.local_index])
        
        # sigmaxz=np.dot(self.Kfv1[:,:,4],self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,4],self.slipv2[self.local_index])
        # sigmayz=np.dot(self.Kfv1[:,:,5],self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,5],self.slipv2[self.local_index])
        # sigmazz=np.dot(self.Kfv1[:,:,2],self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,2],self.slipv2[self.local_index])
        
        #self.plot_stres(sigmaxx)

        self.Atri=[]   
        for i in range(N):
            stres3d=np.identity(3)*trac_nor
            
            self.Tra_vicos[i]=np.dot(stres3d,self.eleVecV[i,-3:])
            P1=np.copy(self.nodelst[self.elelstV[i,0]-1])
            P2=np.copy(self.nodelst[self.elelstV[i,1]-1])
            P3=np.copy(self.nodelst[self.elelstV[i,2]-1])
            A=triangle_area(P1, P2, P3)
            self.Atri.append(A)
            #self.EBforce[i]=self.mu*self.Tra_vicos[i]*A/(1.0/self.vicosity1-1.0/self.vicosity2)
            #self.EBforce[i,0]=
        self.Atri=np.array(self.Atri)
        # self.EBforce[:,0]=sigmaxz*self.Atri*self.mu/self.vicosity
        # self.EBforce[:,1]=sigmayz*self.Atri*self.mu/self.vicosity
        # self.EBforce[:,2]=(sigmazz*2.0/3-sigmaxx/3.0-sigmayy/3.0)*self.Atri*self.mu/self.vicosity
        # #print(rank,np.max(self.EBforce[:,0]),np.max(self.EBforce[:,1]),np.max(self.EBforce[:,2]))

        # self.EBforce=comm.allreduce(self.EBforce, op=MPI.SUM)
        #print(rank,np.max(self.EBforce[:,0]),np.max(self.EBforce[:,1]),np.max(self.EBforce[:,2]))
        

    def dislo_stress(self,obser_points,source_point,fvec):

        X,Y,Z=obser_points[:,0], obser_points[:,1], obser_points[:,2]
        
        n=len(X)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        Z = np.array(Z, dtype=np.float64)
        #print(X)
        Stre=[]
        Ts, Ss, Ds = fvec[2],fvec[0],fvec[1]
        for i in range(len(source_point)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(source_point[i,:3])
            P2=np.copy(source_point[i,3:6])
            P3=np.copy(source_point[i,6:])
            

            stress = np.zeros(n * 6)
            strain = np.zeros(n * 6)
            lib.TDstressFS_C(X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        n,
                        P1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        P2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        P3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        Ss,Ds,Ts,   # Ss, Ds, Ts
                        self.mu_,self.lambda_,       # mu, lambda
                        self.halfspace_jud,
                        stress.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        strain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                        )
            stress=stress.reshape([n,6])
            Stre.append(stress)

        
        return np.array(Stre)
    


    def calc_Kvf(self):

        observesF=np.ascontiguousarray(self.xgF[self.local_index],dtype=np.float64)
        #if(rank==1):
        triangle_verticesV=[]
        
        for i in range(len(self.eleVecV)):
        #for i in range(169,170):
            #print(i,'!!!!!!')
            tri_tem=[]
            P1=np.copy(self.nodelst[self.elelstV[i,0]-1])
            P2=np.copy(self.nodelst[self.elelstV[i,1]-1])
            P3=np.copy(self.nodelst[self.elelstV[i,2]-1])
            tri_tem.append(P1)
            #P2[1]=P2[1]+0.001
            tri_tem.append(P2)
            tri_tem.append(P3)
            
            #f_list.append(fvector)
            tri_tem=np.ascontiguousarray(tri_tem)
            triangle_verticesV.append(tri_tem.reshape([-1]))

        fvector=np.array([1,0,0])
        self.Kvf1=kelvin_stress_lib.compute_kelvin_stresses(observesF,triangle_verticesV,fvector,self.lambda_,self.mu)
        print(rank,' kelvinstres_strike Kvf1 ',self.Kvf1.shape,np.max(self.Kvf1),np.min(self.Kvf1))

        fvector=np.array([0,1,0])
        self.Kvf2=kelvin_stress_lib.compute_kelvin_stresses(observesF,triangle_verticesV,fvector,self.lambda_,self.mu)
        print(rank,' kelvinstres_dip Kvf2',self.Kvf2.shape,np.max(self.Kvf2),np.min(self.Kvf2))

        fvector=np.array([0,0,1])
        self.Kvf3=kelvin_stress_lib.compute_kelvin_stresses(observesF,triangle_verticesV,fvector,self.lambda_,self.mu)
        print(rank,' kelvinstres_dip Kvf3',self.Kvf3.shape,np.max(self.Kvf3),np.min(self.Kvf3))




    def calc_Kvv_kelvinstress(self):
        observesV=np.ascontiguousarray(self.xgV,dtype=np.float64)
        #if(rank==1):
        triangle_verticesV=[]
        
            #tri_tem=[]
        for i in range(len(self.local_indexV)):
        #for i in range(169,170):
            #print(i,'!!!!!!')
            tri_tem=[]
            I=self.local_indexV[i]
            P1=np.copy(self.nodelst[self.elelstV[I,0]-1])
            P2=np.copy(self.nodelst[self.elelstV[I,1]-1])
            P3=np.copy(self.nodelst[self.elelstV[I,2]-1])
            tri_tem.append(P1)
            #P2[1]=P2[1]+0.001
            tri_tem.append(P2)
            tri_tem.append(P3)
            
            #f_list.append(fvector)
            tri_tem=np.ascontiguousarray(tri_tem)
            triangle_verticesV.append(tri_tem.reshape([-1]))
            
        ##VV calculate Kvv,kelvinstresP,dkelvinstresdn
        #f_list=np.ascontiguousarray(f_list,dtype=np.float64)
        triangle_verticesV=np.ascontiguousarray(triangle_verticesV,dtype=np.float64)
        fvector=np.array([1,0,0])
        self.Kvv1=kelvin_stress_lib.compute_kelvin_stresses(observesV,triangle_verticesV,fvector,self.lambda_,self.mu)
        print(rank,' kelvinstres_strike Kvv1 ',self.Kvv1.shape,np.max(self.Kvv1),np.min(self.Kvv1))

        fvector=np.array([0,1,0])
        self.Kvv2=kelvin_stress_lib.compute_kelvin_stresses(observesV,triangle_verticesV,fvector,self.lambda_,self.mu)
        print(rank,' kelvinstres_dip Kvv2 ',self.Kvv2.shape,np.max(self.Kvv2),np.min(self.Kvv2))

        fvector=np.array([0,0,1])
        self.Kvv3=kelvin_stress_lib.compute_kelvin_stresses(observesV,triangle_verticesV,fvector,self.lambda_,self.mu)
        print(rank,' kelvinstres_nro Kvv3 ',self.Kvv3.shape,np.max(self.Kvv3),np.min(self.Kvv3))


        self.kelvinstresP = kelvin_stress_potential_lib.compute_stresses_potential(observesV,triangle_verticesV,self.lambda_,self.mu,mode=0)
        
        print(rank,' kelvinstresP ',self.kelvinstresP.shape,np.max(self.kelvinstresP),np.min(self.kelvinstresP))
        #kelvinstresP = np.sum(kelvinstresP, axis=0)
        
        self.dkelvinstresdn = kelvin_stress_potential_lib.compute_stresses_potential(observesV,triangle_verticesV,self.lambda_,self.mu,mode=1)
        np.save('dkelvinstresdn',self.dkelvinstresdn)
        #dkelvinstresdn = np.sum(dkelvinstresdn, axis=0)
        print(rank,' dkelvinstresdn ',self.dkelvinstresdn.shape,np.max(self.dkelvinstresdn),np.min(self.dkelvinstresdn))
       
    

    def calc_Pre_potential(self):
        observesV=np.ascontiguousarray(self.xgV,dtype=np.float64)
        triangle_verticesV=[]
        if (rank==0):
            for I in range(len(self.elelstV)):
                tri_tem=[]
                P1=np.copy(self.nodelst[self.elelstV[I,0]-1])
                P2=np.copy(self.nodelst[self.elelstV[I,1]-1])
                P3=np.copy(self.nodelst[self.elelstV[I,2]-1])
                tri_tem.append(P1)
                #P2[1]=P2[1]+0.001
                tri_tem.append(P2)
                tri_tem.append(P3)
                
                #f_list.append(fvector)
                tri_tem=np.ascontiguousarray(tri_tem)
                triangle_verticesV.append(tri_tem.flatten())
                
            self.pres_potential = laplace_lib.compute_pressure_potential(observesV,triangle_verticesV,mode=0)
            print(rank,' pres_potential ',self.pres_potential.shape,np.max(self.pres_potential),np.min(self.pres_potential))
            #self.pres_potential = np.sum(pres_potential, axis=0)
            self.dpres_potentialdn = laplace_lib.compute_pressure_potential(observesV,triangle_verticesV,mode=1)
            print(rank,' dpres_potentialdn ',self.dpres_potentialdn.shape,np.max(self.dpres_potentialdn),np.min(self.dpres_potentialdn))
            #self.dpres_potentialdn = np.sum(dpres_potentialdn, axis=0)
            


    def init_mpi_local_visco_variables(self):
        self.local_indexV=self.parallel_cells_scatter_send(N=len(self.elelstV))
        self.countsV = comm.gather(len(self.local_indexV), root=0)
        self.displsV = comm.gather(self.local_indexV[0], root=0)
        self.sigmaV=np.zeros([len(self.eleVecV),6])
        self.Pv0=(self.sigmaV[:,0]+self.sigmaV[:,1]+self.sigmaV[:,2])/3.0
        # self.sigmaV=np.ones([len(self.local_indexV),6])
        

        self.calc_Kvv_kelvinstress()
        self.calc_Kvf()
        self.calc_Pre_potential()



        self.Kfv1,self.Kfv2=self.calc_corefunc_disloc()


        #calculate pres_potential and dpres_potentialdn
        

        
        self.compute_volum_integral_term()
        self.compute_surface_integral_term()
    

    def stress_surf_dot_vectorized(self, stress_vec, elevec,devot=False):
        """
        向量化加速版本的张量收缩计算
        stress_vec 形状: (N, 6) -> 分量分别为: s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
        self.eleVec 形状: (N, 维度)，后三列 [-3:] 为法向量 n_x, n_y, n_z
        """
        if(devot==True):
            p = (stress_vec[:, 0] + stress_vec[:, 1] + stress_vec[:, 2]) / 3.0
        else:
            p=0
        # 2. 批量构建偏应力张量的 6 个独立分量，形状均为 (N,)
        s_xx_prime = stress_vec[:, 0] - p
        s_yy_prime = stress_vec[:, 1] - p
        s_z_prime = stress_vec[:, 2] - p
        s_xy = stress_vec[:, 3]
        s_xz = stress_vec[:, 4]
        s_yz = stress_vec[:, 5]

        # 3. 组合成三维应力矩阵阵列，形状为 (N, 3, 3)
        N = len(stress_vec)
        stress3D = np.empty((N, 3, 3), dtype=stress_vec.dtype)

        stress3D[:, 0, 0] = s_xx_prime
        stress3D[:, 0, 1] = s_xy
        stress3D[:, 0, 2] = s_xz
        stress3D[:, 1, 0] = s_xy
        stress3D[:, 1, 1] = s_yy_prime
        stress3D[:, 1, 2] = s_yz
        stress3D[:, 2, 0] = s_xz
        stress3D[:, 2, 1] = s_yz
        stress3D[:, 2, 2] = s_z_prime

        # 4. 提取法向量，形状为 (N, 3)
        normals = elevec[:, -3:]
        

        # 5. 使用 einsum 进行批量矩阵-向量乘法 (相当于对每个单元执行 np.dot)
        # 'nij,nj->ni' 表示: 对第 n 个单元，(3,3)矩阵的第 j 维 与 (3,)向量的第 j 维收缩，输出 (3,) 向量
        traction = np.einsum("nij,nj->ni", stress3D, normals)

        return traction
    

    def stress_surf_dot(self,stress_vec,elevec,devot=False):
        Traction=[]
        for i in range(len(stress_vec)):
            if(devot==True):
                p=(stress_vec[i,0]+stress_vec[i,1]+stress_vec[i,2])/3.0
            else:
                p=0
            stress3D=np.array([[stress_vec[i,0]-p,stress_vec[i,3],stress_vec[i,4]],
                               [stress_vec[i,3],stress_vec[i,1]-p,stress_vec[i,5]],
                               [stress_vec[i,4],stress_vec[i,5],stress_vec[i,2]-p]])
            tra=np.dot(stress3D,elevec[i,-3:])
            Traction.append(tra)
            # if(i==0):
            #     print(stress3D,tra,elevec[i,-3:])
        return np.array(Traction)

    def compute_volum_integral_term(self):

        self.Pv0=(self.sigmaV[:,0]+self.sigmaV[:,1]+self.sigmaV[:,2])/3.0
        Ivol=None
        dPdn=None
        if(rank==0):
            weightp= np.linalg.solve(self.pres_potential.transpose(), self.Pv0)
            
            dPdn=np.dot(self.dpres_potentialdn.transpose(),weightp)
            #print('dPdn ',dPdn.shape,np.max(dPdn))

        
        dPdn_bcast=comm.bcast(dPdn, root=0)
        self.dSPdn_dot_P_reduce = np.zeros([len(self.elelstV),6], dtype=np.float64)
        self.SP_dot_dPdn_reduce  = np.zeros([len(self.elelstV),6], dtype=np.float64)
        
        dSPdn_dot_P = self.dkelvinstresdn * self.Pv0[self.local_indexV, None, None]
        dSPdn_dot_P = np.sum(dSPdn_dot_P, axis=0)
        # dSPdn_dot_P=[]
        # for i in range(6):
        #     tem=np.dot(self.dkelvinstresdn[:,:,i].transpose(),self.Pv0[self.local_indexV])
        #     dSPdn_dot_P.append(tem)
        # dSPdn_dot_P=np.array(dSPdn_dot_P).transpose()
        #np.save(f'dSPdn_dot_P{rank}',dSPdn_dot_P)
        
        
        SP_dot_dPdn = self.kelvinstresP * dPdn_bcast[self.local_indexV, None, None]
        SP_dot_dPdn = np.sum(SP_dot_dPdn, axis=0)

        comm.Reduce(dSPdn_dot_P, self.dSPdn_dot_P_reduce, op=MPI.SUM, root=0)
        comm.Reduce(SP_dot_dPdn, self.SP_dot_dPdn_reduce, op=MPI.SUM, root=0)

        # else:
        #     self.kelvinstresP_reduce = None
        #     self.dkelvinstresdn_reduce = None
        # comm.Reduce(kelvinstresP, self.kelvinstresP_reduce, op=MPI.SUM, root=0)
        # comm.Reduce(dkelvinstresdn, self.dkelvinstresdn_reduce, op=MPI.SUM, root=0)

        
        
        nor_dot_Pres=self.eleVecV[:,-3:]* self.Pv0[:, np.newaxis]
        Gn1=np.dot(self.Kvv1[:,:,0].transpose(),nor_dot_Pres[self.local_indexV,0])+np.dot(self.Kvv2[:,:,0].transpose(),nor_dot_Pres[self.local_indexV,1])\
            +np.dot(self.Kvv3[:,:,0].transpose(),nor_dot_Pres[self.local_indexV,2])
        Gn2=np.dot(self.Kvv1[:,:,1].transpose(),nor_dot_Pres[self.local_indexV,0])+np.dot(self.Kvv2[:,:,1].transpose(),nor_dot_Pres[self.local_indexV,1])\
            +np.dot(self.Kvv3[:,:,1].transpose(),nor_dot_Pres[self.local_indexV,2])
        Gn3=np.dot(self.Kvv1[:,:,2].transpose(),nor_dot_Pres[self.local_indexV,0])+np.dot(self.Kvv2[:,:,2].transpose(),nor_dot_Pres[self.local_indexV,1])\
            +np.dot(self.Kvv3[:,:,2].transpose(),nor_dot_Pres[self.local_indexV,2])
        Gn4=np.dot(self.Kvv1[:,:,3].transpose(),nor_dot_Pres[self.local_indexV,0])+np.dot(self.Kvv2[:,:,3].transpose(),nor_dot_Pres[self.local_indexV,1])\
            +np.dot(self.Kvv3[:,:,3].transpose(),nor_dot_Pres[self.local_indexV,2])
        Gn5=np.dot(self.Kvv1[:,:,4].transpose(),nor_dot_Pres[self.local_indexV,0])+np.dot(self.Kvv2[:,:,4].transpose(),nor_dot_Pres[self.local_indexV,1])\
            +np.dot(self.Kvv3[:,:,4].transpose(),nor_dot_Pres[self.local_indexV,2])
        Gn6=np.dot(self.Kvv1[:,:,5].transpose(),nor_dot_Pres[self.local_indexV,0])+np.dot(self.Kvv2[:,:,5].transpose(),nor_dot_Pres[self.local_indexV,1])\
            +np.dot(self.Kvv3[:,:,5].transpose(),nor_dot_Pres[self.local_indexV,2])
        
        Gn1_reduce = np.empty_like(Gn1)
        Gn2_reduce = np.empty_like(Gn2)
        Gn3_reduce = np.empty_like(Gn3)
        Gn4_reduce = np.empty_like(Gn4)
        Gn5_reduce = np.empty_like(Gn5)
        Gn6_reduce = np.empty_like(Gn6)
        
        comm.Reduce(Gn1,Gn1_reduce, op=MPI.SUM, root=0)
        comm.Reduce(Gn2,Gn2_reduce, op=MPI.SUM, root=0)
        comm.Reduce(Gn3,Gn3_reduce, op=MPI.SUM, root=0)
        comm.Reduce(Gn4,Gn4_reduce, op=MPI.SUM, root=0)
        comm.Reduce(Gn5,Gn5_reduce, op=MPI.SUM, root=0)
        comm.Reduce(Gn6,Gn6_reduce, op=MPI.SUM, root=0)

        if(rank==0):
            self.Gn_P=np.array([Gn1_reduce,Gn2_reduce,Gn3_reduce,Gn4_reduce,Gn5_reduce,Gn6_reduce]).transpose()

            beta=1.0/self.vicosity*self.mu
            self.Ivol=(self.dSPdn_dot_P_reduce-self.Gn_P)*beta-self.SP_dot_dPdn_reduce*beta

            print('self.dSPdn_dot_P_reduce ',np.max(self.dSPdn_dot_P_reduce),np.min(self.dSPdn_dot_P_reduce))
            print('self.Gn_P ',np.max(self.Gn_P),np.min(self.Gn_P))
            print('Ivol ',self.Ivol.shape,np.max(self.Ivol))
        

        
            

        


    def compute_surface_integral_term(self):
        #print(self.sigmaV.shape,self.eleVecV[self.local_indexV].shape)
        tracS=self.stress_surf_dot(self.sigmaV[self.local_indexV],self.eleVecV[self.local_indexV],devot=True)
        delta_beta=self.mu/self.vicosity0-self.mu/self.vicosity
        tracS=tracS* delta_beta
        #print("tracS: ",tracS.shape,np.min(tracS))
        # tracS[:,0]=self.sigmaV[self.local_indexV,4]*self.mu/self.vicosity
        # #print(np.min((self.mu/self.vicosity0-self.mu/self.vicosity)))
        # tracS[:,1]=self.sigmaV[self.local_indexV,5]*self.mu/self.vicosity
        # tracS[:,2]=(self.sigmaV[self.local_indexV,2]*2.0/3-self.sigmaV[self.local_indexV,0]/3.0-self.sigmaV[self.local_indexV,1]/3.0)*self.mu/self.vicosity
        
        dsigmaV1=np.dot(self.Kvv1[:,:,0].transpose(),tracS[:,0])+np.dot(self.Kvv2[:,:,0].transpose(),tracS[:,1])+np.dot(self.Kvv3[:,:,0].transpose(),tracS[:,2])
        dsigmaV2=np.dot(self.Kvv1[:,:,1].transpose(),tracS[:,0])+np.dot(self.Kvv2[:,:,1].transpose(),tracS[:,1])+np.dot(self.Kvv3[:,:,1].transpose(),tracS[:,2])
        dsigmaV3=np.dot(self.Kvv1[:,:,2].transpose(),tracS[:,0])+np.dot(self.Kvv2[:,:,2].transpose(),tracS[:,1])+np.dot(self.Kvv3[:,:,2].transpose(),tracS[:,2])
        dsigmaV4=np.dot(self.Kvv1[:,:,3].transpose(),tracS[:,0])+np.dot(self.Kvv2[:,:,3].transpose(),tracS[:,1])+np.dot(self.Kvv3[:,:,3].transpose(),tracS[:,2])
        dsigmaV5=np.dot(self.Kvv1[:,:,4].transpose(),tracS[:,0])+np.dot(self.Kvv2[:,:,4].transpose(),tracS[:,1])+np.dot(self.Kvv3[:,:,4].transpose(),tracS[:,2])
        dsigmaV6=np.dot(self.Kvv1[:,:,5].transpose(),tracS[:,0])+np.dot(self.Kvv2[:,:,5].transpose(),tracS[:,1])+np.dot(self.Kvv3[:,:,5].transpose(),tracS[:,2])


        Gn1=comm.allreduce(dsigmaV1.flatten(), op=MPI.SUM)
        Gn2=comm.allreduce(dsigmaV2.flatten(), op=MPI.SUM)
        Gn3=comm.allreduce(dsigmaV3.flatten(), op=MPI.SUM)
        Gn4=comm.allreduce(dsigmaV4.flatten(), op=MPI.SUM)
        Gn5=comm.allreduce(dsigmaV5.flatten(), op=MPI.SUM)
        Gn6=comm.allreduce(dsigmaV6.flatten(), op=MPI.SUM)

        self.Isurf=np.array([Gn1,Gn2,Gn3,Gn4,Gn5,Gn6]).transpose()
        if(rank==0):
            print(rank, ' Isurf ',self.Isurf.shape,np.max(self.Isurf),np.min(self.Isurf))


        tracS=self.stress_surf_dot(self.sigmaV,self.eleVecV)
        dsigmaV1=np.dot(self.Kvf1[:,:,0].transpose(),tracS[:,0])+np.dot(self.Kvf2[:,:,0].transpose(),tracS[:,1])+np.dot(self.Kvf3[:,:,0].transpose(),tracS[:,2])
        dsigmaV2=np.dot(self.Kvf1[:,:,1].transpose(),tracS[:,0])+np.dot(self.Kvf2[:,:,1].transpose(),tracS[:,1])+np.dot(self.Kvf3[:,:,1].transpose(),tracS[:,2])
        dsigmaV3=np.dot(self.Kvf1[:,:,2].transpose(),tracS[:,0])+np.dot(self.Kvf2[:,:,2].transpose(),tracS[:,1])+np.dot(self.Kvf3[:,:,2].transpose(),tracS[:,2])
        dsigmaV4=np.dot(self.Kvf1[:,:,3].transpose(),tracS[:,0])+np.dot(self.Kvf2[:,:,3].transpose(),tracS[:,1])+np.dot(self.Kvf3[:,:,3].transpose(),tracS[:,2])
        dsigmaV5=np.dot(self.Kvf1[:,:,4].transpose(),tracS[:,0])+np.dot(self.Kvf2[:,:,4].transpose(),tracS[:,1])+np.dot(self.Kvf3[:,:,4].transpose(),tracS[:,2])
        dsigmaV6=np.dot(self.Kvf1[:,:,5].transpose(),tracS[:,0])+np.dot(self.Kvf2[:,:,5].transpose(),tracS[:,1])+np.dot(self.Kvf3[:,:,5].transpose(),tracS[:,2])

        

        e11,e12,e13=self.eleVecF[self.local_index,0],self.eleVecF[self.local_index,1],self.eleVecF[self.local_index,2]
        e21,e22,e23=self.eleVecF[self.local_index,3],self.eleVecF[self.local_index,4],self.eleVecF[self.local_index,5]
        e31,e32,e33=self.eleVecF[self.local_index,6],self.eleVecF[self.local_index,7],self.eleVecF[self.local_index,8]



        self.VFdot1_local=dsigmaV1*e31*e11+dsigmaV4*e32*e12+dsigmaV5*e33*e13
        self.VFdot2_local=dsigmaV4*e31*e21+dsigmaV2*e32*e22+dsigmaV6*e33*e23
        self.VFdot3_local=dsigmaV5*e31*e11+dsigmaV6*e32*e12+dsigmaV3*e33*e13

        #print('self.VFdot1_local',self.VFdot1_local.shape,np.max(self.VFdot1_local))
        # self.VFdot2_local=np.dot(self.Kvf1[:,:,1],self.EBforce[:,0])+np.dot(self.Kvf2[:,:,1],self.EBforce[:,1])+np.dot(self.Kvf3[:,:,1],self.EBforce[:,2])
        # self.VFdot3_local=np.zeros(len(self.VFdot2_local))
        # if(self.fix_Tn==False):
        #     self.VFdot3_local=np.dot(self.Kvf1[:,:,2],self.EBforce[:,0])+np.dot(self.Kvf2[:,:,2],self.EBforce[:,1])+np.dot(self.Kvf3[:,:,2],self.EBforce[:,2])






    def updata_sigmaV(self,dt=0):
        
        Nv=len(self.elelstV)
        dsigmaxx_fromF0=np.zeros(Nv)
        dsigmayy_fromF0=np.zeros(Nv)
        dsigmazz_fromF0=np.zeros(Nv)
        dsigmaxy_fromF0=np.zeros(Nv)
        dsigmaxz_fromF0=np.zeros(Nv)
        dsigmayz_fromF0=np.zeros(Nv)

        dsigmaxx_fromF=np.dot(self.Kfv1[:,:,0].transpose(),self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,0].transpose(),self.slipv2[self.local_index])
        dsigmayy_fromF=np.dot(self.Kfv1[:,:,1].transpose(),self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,1].transpose(),self.slipv2[self.local_index])
        dsigmazz_fromF=np.dot(self.Kfv1[:,:,2].transpose(),self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,2].transpose(),self.slipv2[self.local_index])
        dsigmaxy_fromF=np.dot(self.Kfv1[:,:,3].transpose(),self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,3].transpose(),self.slipv2[self.local_index])
        dsigmaxz_fromF=np.dot(self.Kfv1[:,:,4].transpose(),self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,4].transpose(),self.slipv2[self.local_index])
        dsigmayz_fromF=np.dot(self.Kfv1[:,:,5].transpose(),self.slipv1[self.local_index])+np.dot(self.Kfv2[:,:,5].transpose(),self.slipv2[self.local_index])
        
        
        
        comm.Reduce(dsigmaxx_fromF, dsigmaxx_fromF0, op=MPI.SUM, root=0)
        comm.Reduce(dsigmayy_fromF, dsigmayy_fromF0, op=MPI.SUM, root=0)
        comm.Reduce(dsigmazz_fromF, dsigmazz_fromF0, op=MPI.SUM, root=0)
        comm.Reduce(dsigmaxy_fromF, dsigmaxy_fromF0, op=MPI.SUM, root=0)
        comm.Reduce(dsigmaxz_fromF, dsigmaxz_fromF0, op=MPI.SUM, root=0)
        comm.Reduce(dsigmayz_fromF, dsigmayz_fromF0, op=MPI.SUM, root=0)


        

        if(rank==0):
            Pressure=(self.sigmaV[:,2]/3+self.sigmaV[:,0]/3.0+self.sigmaV[:,1]/3.0)
            #self.dsigma_fromF0=np.array([dsigmaxx_fromF0,dsigmayy_fromF0,dsigmazz_fromF0,dsigmaxy_fromF0,dsigmaxz_fromF0,dsigmayz_fromF0]).transpose()
            #self.sigmaV=self.sigmaV+(self.dsigma_fromF0+self.Isurf)*dt
            #self.sigmaV_rate=(self.dsigma_fromF0+self.Isurf)

            dpres_fromF0=dsigmaxx_fromF0/3.0+dsigmayy_fromF0/3.0+dsigmazz_fromF0/3.0

            
            self.sigmaV[:,0]=self.sigmaV[:,0]+(-self.Isurf[:,0]+dsigmaxx_fromF0-dpres_fromF0-self.Ivol[:,0]-self.mu/self.vicosity*(self.sigmaV[:,0]-Pressure))*dt
            self.sigmaV[:,1]=self.sigmaV[:,1]+(-self.Isurf[:,0]+dsigmayy_fromF0-dpres_fromF0-self.Ivol[:,1]-self.mu/self.vicosity*(self.sigmaV[:,1]-Pressure))*dt
            self.sigmaV[:,2]=self.sigmaV[:,2]+(-self.Isurf[:,0]+dsigmazz_fromF0-dpres_fromF0-self.Ivol[:,2]-self.mu/self.vicosity*(self.sigmaV[:,2]-Pressure))*dt
            self.sigmaV[:,3]=self.sigmaV[:,3]+(-self.Isurf[:,0]+dsigmaxy_fromF0-self.Ivol[:,3]-self.mu/self.vicosity*self.sigmaV[:,3])*dt
            self.sigmaV[:,4]=self.sigmaV[:,4]+(-self.Isurf[:,0]+dsigmaxz_fromF0-self.Ivol[:,4]-self.mu/self.vicosity*self.sigmaV[:,4])*dt
            self.sigmaV[:,5]=self.sigmaV[:,5]+(-self.Isurf[:,0]+dsigmayz_fromF0-self.Ivol[:,5]-self.mu/self.vicosity*self.sigmaV[:,5])*dt
            self.sigmaV_rate=-self.Isurf[:,0]+dsigmaxx_fromF0-dpres_fromF0-self.Ivol[:,0]-self.mu/self.vicosity*(self.sigmaV[:,0]-Pressure)

            # self.sigmaV=self.sigmaV+(self.Isurf)*dt
            # self.sigmaV_rate=self.Isurf
            #print('self.dsigma_fromF0 ',np.max(self.dsigma_fromF0[:,3]),np.min(self.dsigma_fromF0[:,3]))
            #print('self.sigmaV_rate ',np.max(self.sigmaV_rate[:,3]),np.min(self.sigmaV_rate[:,3]))
            print('self.Isurf ',np.max(self.Isurf[:,3]),np.min(self.Isurf[:,3]))
            print('self.Ivol ',np.max(self.Ivol[:,3]),np.min(self.Ivol[:,3]))
            #print(self.dsigma_fromF0[1000,3],self.sigmaV_rate[1000,3],self.Isurf[1000,3],self.sigmaV[1000,3])
            
            #print('updated sigmaV ',np.max(self.sigmaV[:,3]),np.min(self.sigmaV[:,3]))


            Val = [
                np.mean(np.abs(self.sigmaV[:,0])),
                np.mean(np.abs(self.sigmaV[:,1])),
                np.mean(np.abs(self.sigmaV[:,2])),
                np.mean(np.abs(self.sigmaV[:,3])),
                np.mean(np.abs(self.sigmaV[:,4])),
                np.mean(np.abs(self.sigmaV[:,5]))
            ]


            print('sigmaV: ',Val)


        comm.Bcast(self.sigmaV, root=0)
        self.sigmaV=comm.bcast(self.sigmaV, root=0)
        
    

    

    def start(self):
        start_time = MPI.Wtime()
        totaloutputsteps=int(self.Para0['totaloutputsteps']) #total time steps
        file = open(self.state_file, "a", encoding="utf-8")
        SLIPV=[]
        Tt=[]
        #self.Init_mpi_local_variables()
        self.init_mpi_local_variables()
        #self.init_vicosity_stress()
        file.write('iteration time_step(s) maximum_slip1_rate(m/s) maximum_slip2_rate(m/s) time(s) time(d)\n')
        for i in range(totaloutputsteps):
        #for i in range(0):
            self.step=i
            if(i==0):#inital step length
                dttry=self.htry
            else:
                dttry=dtnext
            dttry,dtnext=self.simu_forward_mpi(dttry)

            if(dtnext>1e8  and i>1000):
                dtnext=1e8

            if(rank==0):
                self.output_visco()
                year=self.time/3600/24/365
                #if(i%10==0):
                print('iteration:',i, flush=True)
                print('dt:',dttry,' max_vel:',np.max(np.abs(self.slipv)),' min_vel:',np.min(np.abs(self.slipv)),' Porepressure max:',np.max(self.P),' Porepressure min:',np.min(self.P),' dpdt_max:',np.max((self.dPdt0)),' dpdt_min:',np.min((self.dPdt0)),' Seconds:',self.time,'  Days:',self.time/3600/24,
                'year',year, flush=True)
                #Output screen information: Iteration; time step; slipv1; slipv2; second; hours
                file.write('%d %f %.16f %.16e %f %f\n' %(i,dttry,np.max(np.abs(self.slipv1)),np.max(np.abs(self.slipv2)),self.time,self.time/3600.0/24.0))
                file.flush()
                #f1.write('%d %f %f %f %.6e %.16e\n'%(i,dttry,sim0.time,sim0.time/3600.0/24.0,sim0.Tt[index1_],sim0.slipv[index1_]))
                #SLIP.append(sim0.slip)

                #Save slip rate and shear stress for each iteration
                SLIPV.append(self.slipv)
                #Tt.append(self.Tt)
                
                # if(sim0.time>60):
                #     break
                #Output vtk once every outsteps
                outsteps=int(self.Para0['outsteps'])
                directory='out_vtu'
                if not os.path.exists(directory):
                    os.mkdir(directory)
                #output slipv and Tt
                if(i%outsteps==0):
                    #SLIP=np.array(SLIP)
                    SLIPV=np.array(SLIPV)
                    Tt=np.array(Tt)
                    if(self.Para0['outputSLIPV']==True):
                        directory1='out_slipvTt'
                        if not os.path.exists(directory1):
                            os.mkdir(directory1)
                        np.save(directory1+'/slipv_%d'%i,SLIPV)
                    # if(self.Para0['outputTt']==True):
                    #     directory1='out_slipvTt'
                    #     if not os.path.exists(directory1):
                    #         os.mkdir(directory1)
                    #     np.save(directory1+'/Tt_%d'%i,Tt)


                    #SLIP=[]
                    SLIPV=[]
                    #Tt=[]
                    #output vtk
                    if(self.Para0['outputvtu']==True):
                        #print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                        fname=directory+'/step'+str(i)+'.vtu'
                        self.writeVTU(fname)
                        #self.output_visco()
                    # if(self.Para0['outputmatrix']==True):
                    #     fname='step'+str(i)
                    #     self.writeVTU(fname)
                   

        end_time = MPI.Wtime()
        #print(f"rank {rank} computation time: {self.compute_time:.6f} sec")
        if rank == 0:
            print(f"Program run time: {end_time - start_time:.6f} sec")
            print(f"communication time: {self.comm_time:.6f} sec")
            print(f"Matrix product computation time: {self.compute_time:.6f} sec")
            timetake=end_time - start_time
            file.write(f"Program run time: {end_time - start_time:.6f} sec")
            file.write(f"communication time: {self.comm_time:.6f} sec")
            file.write(f"Matrix product computation time: {self.compute_time:.6f} sec")
            file.write('Program end time: %s\n'%str(datetime.now()))
            #file.write("Time taken: %.2f seconds\n"%timetake)
            file.close()
            #print('menmorary:',s*6)



    def simu_forward_mpi(self,dttry):
        #print('self.P',np.max(self.P))
        if(self.step<1000):
            slipv1=self.slipv1-self.slipvC*np.cos(self.rake0)
            slipv2=self.slipv2-self.slipvC*np.sin(self.rake0)
        else:
            slipv1=self.slipv1
            slipv2=self.slipv2
        #Calculating Kv first
        #comm.Barrier()
        t0 = MPI.Wtime()
        if(self.fix_Tn==True):
            dsigmadt=self.normal_loading
        else:
            #self.Tno=comm.bcast(self.Tno, root=0)
            #dsigmadt=np.dot(self.Bs,slipv1)+np.dot(self.Bd,slipv2)+self.normal_loading
            dsigmadt=self.tree_block.blocks_process_MVM(slipv1,self.local_blocks,'Bs')+\
                self.tree_block.blocks_process_MVM(slipv2,self.local_blocks,'Bd')+self.normal_loading

        #dsigmadt[self.index_normal]=-dsigmadt[self.index_normal]
        AdotV1=self.tree_block.blocks_process_MVM(slipv1,self.local_blocks,'A1s')+\
                self.tree_block.blocks_process_MVM(slipv2,self.local_blocks,'A1d')
        AdotV2=self.tree_block.blocks_process_MVM(slipv1,self.local_blocks,'A2s')+\
                self.tree_block.blocks_process_MVM(slipv2,self.local_blocks,'A2d')

        
        
        #print(AdotV1,AdotV2)
        t1 = MPI.Wtime()
        self.compute_time += (t1 - t0)

        #Combine results from all ranks
        t0 = MPI.Wtime()
        self.dsigmadt=comm.allreduce(dsigmadt, op=MPI.SUM)
        self.AdotV1=comm.allreduce(AdotV1, op=MPI.SUM)
        self.AdotV2=comm.allreduce(AdotV2, op=MPI.SUM)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)

        nrjct=0
        h=dttry
        running=True
        dtnext=None

        while running:
            Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk=self.RungeKutte_solve_Dormand_Prince_(h)
            global_Relerrormax1 = comm.allreduce(self.Relerrormax1, op=MPI.MAX)
            global_Relerrormax2 = comm.allreduce(self.Relerrormax2, op=MPI.MAX)
            # global_Relerrormax1=comm.bcast(global_Relerrormax1, root=0)
            # global_Relerrormax2=comm.bcast(global_Relerrormax2, root=0)
            self.RelTol1=1e-4
            self.RelTol2=1e-4
            condition1=global_Relerrormax1/self.RelTol1
            condition2=global_Relerrormax2/self.RelTol2
            hnew1=h*0.9*(self.RelTol1/global_Relerrormax1)**0.2
            hnew2=h*0.9*(self.RelTol2/global_Relerrormax2)**0.2
            #print(hnew1,hnew2)
            
            if(max(condition1,condition2)<1.0 and not (np.isnan(condition1) or np.isnan(condition2))):
                #print(type(hnew1),type(condition1))
                dtnext=min(hnew1,hnew2)
                dtnext=min(1.5*h,dtnext)
                break
                
                
            else:
                nrjct=nrjct+1
                dtnext=min(hnew1,hnew2)
                h=max(0.5*h,dtnext)
                #h=0.5*h
                #print('nrjct:',nrjct,'  condition1,',condition1,' condition2:',condition2,'  dt:',h)

                if(h<1.e-15 or nrjct>20):
                    print('error: dt is too small')
                    sys.exit()

        self.time=self.time+h

        #if(rank==0):
        #update slip rate and rake
        Tno_yhk[Tno_yhk<0.1]=0.1
        self.Tno_local=Tno_yhk
        self.Tt1o_local=Tt1o_yhk
        self.Tt2o_local=Tt2o_yhk
        self.state_local=state_yhk
        #print(np.max(self.Tt1o_local),np.max(self.state_local),rank)
        
        #self.Tt_local=np.sqrt(Tt1o_yhk*Tt1o_yhk+Tt2o_yhk*Tt2o_yhk)
        #print('self.Tt1o',np.mean(self.Tt1o),np.mean(self.Tt2o))
        self.slipv1[:]=0
        self.slipv2[:]=0
        self.slipv1[self.local_index]=(2.0*self.V0)*np.exp(-self.state_local/self.a[self.local_index])*np.sinh(self.Tt1o_local/(self.Tno_local-self.P[self.local_index]*1e-6)/self.a[self.local_index])
        self.slipv2[self.local_index]=(2.0*self.V0)*np.exp(-self.state_local/self.a[self.local_index])*np.sinh(self.Tt2o_local/(self.Tno_local-self.P[self.local_index]*1e-6)/self.a[self.local_index])
        #print(np.max(np.exp(-self.state_local/self.a[self.local_index])),rank)
        t0 = MPI.Wtime()
        self.slipv1=comm.allreduce(self.slipv1, op=MPI.SUM)
        self.slipv2=comm.allreduce(self.slipv2, op=MPI.SUM)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)
        self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)

        #print(np.max(self.slipv))
        #self.rake=np.arctan2(self.Tt2o,self.Tt1o)
        
        indexmin=np.where(self.slipv<1e-30)[0]
        if(len(indexmin)>0):
            self.slipv[indexmin]=1e-30
        #self.maxslipv0=np.max(self.slipv)
        #update slip
        self.slip1=self.slip1+self.slipv1*h
        self.slip2=self.slip2+self.slipv2*h
        self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)

        self.compute_volum_integral_term()
        self.compute_surface_integral_term()
        self.updata_sigmaV(h)
        

        if(self.step%self.Para0['outsteps']==0):
            #print(self.counts, self.displs,self.Tno.shape,Tno_yhk.shape)
            #print(Tno_yhk.dtype, self.Tno.dtype)
            t0 = MPI.Wtime()
            comm.Gatherv(sendbuf=Tno_yhk,recvbuf=(self.Tno, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=Tt1o_yhk,recvbuf=(self.Tt1o, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=Tt2o_yhk,recvbuf=(self.Tt2o, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=state_yhk,recvbuf=(self.state, (self.counts, self.displs)), root=0)

            comm.Gatherv(sendbuf=self.VFdot1_local,recvbuf=(self.VFdot1, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=self.VFdot2_local,recvbuf=(self.VFdot2, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=self.VFdot3_local,recvbuf=(self.VFdot3, (self.counts, self.displs)), root=0)
            
            
            if(self.Ifdila==True):
                self.porosity[self.index_]=0
                recvbuf = np.zeros(len(self.porosity), dtype=np.float64) 
                comm.Reduce(self.porosity, recvbuf, op=MPI.SUM, root=0)
                if(rank==0):
                    self.porosity=recvbuf
            
            t1 = MPI.Wtime()
            self.comm_time += (t1 - t0)
            if(rank==0):
                self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
                self.rake=np.arctan2(self.slip2,self.slip1)
                self.fric=self.Tt/(self.Tno-self.P*1e-6)

                

            if(self.Ifdila==False):
                t0 = MPI.Wtime()
                comm.Gatherv(sendbuf=state_yhk,recvbuf=(self.state, (self.counts, self.displs)), root=0)
                t1 = MPI.Wtime()
                self.comm_time += (t1 - t0)
            
        #update temperature
        if(self.Ifthermal==True):
            Tempe,dTdt0,Tarr=self.Calc_T_implicit_mpi(h)
            self.Tempe=Tempe
            self.dTdt0=dTdt0
            self.Tempearr=Tarr
            if(self.step%self.Para0['outsteps']==0):
                Tarr[self.index_]=0
                Tempe[self.index_]=0
                dTdt0[self.index_]=0
                self.dTdt0=comm.allreduce(dTdt0, op=MPI.SUM)
                self.Tempe=comm.allreduce(Tempe, op=MPI.SUM)
                self.Tempearr=comm.allreduce(Tarr, op=MPI.SUM)

        #update Pore pressure

        if(self.Ifdila==True):
            #comm.Allgatherv(sendbuf=state_yhk,recvbuf=(self.state, (self.counts, self.displs)))
            Pre,dPdt0,Parr=self.Calc_P_implicit_mpi(h)
            
            self.dPdt0=dPdt0
            self.P=Pre
            self.Parr=Parr
            if(self.step%self.Para0['outsteps']==0):
                Parr[self.index_]=0
                Pre[self.index_]=0
                dPdt0[self.index_]=0
                self.dPdt0=comm.allreduce(dPdt0, op=MPI.SUM)
                self.P=comm.allreduce(Pre, op=MPI.SUM)
                self.Parr=comm.allreduce(Parr, op=MPI.SUM)

        

        return h,dtnext






    def derivative_(self,Tno,Tt1o,Tt2o,state):
        Tno=Tno*1e6
        Tt1o=Tt1o*1e6
        Tt2o=Tt2o*1e6
        
        P=self.P[self.local_index]
        dPdt=self.dPdt0[self.local_index]
        AdotV1=self.AdotV1[self.local_index]+self.VFdot1_local
        AdotV2=self.AdotV2[self.local_index]+self.VFdot2_local
        dsigmadt=self.dsigmadt[self.local_index]+self.VFdot3_local
        # AdotV1=self.AdotV1[self.local_index]
        # AdotV2=self.AdotV2[self.local_index]
        # dsigmadt=self.dsigmadt[self.local_index]
        shear_loading=self.shear_loading[self.local_index]
        
        slipv=self.slipv[self.local_index]

        def safe_exp(x, max_value=700):  # 限制指数的最大值
            return np.exp(np.clip(x, -max_value, max_value))

        def safe_cosh(x, max_value=700):  # 使用指数形式的cosh，避免溢出
            x = np.clip(x, -max_value, max_value)
            return (np.exp(x) + np.exp(-x)) / 2

        def safe_sinh(x, max_value=700):  # 使用指数形式的sinh，避免溢出
            x = np.clip(x, -max_value, max_value)
            return (np.exp(x) - np.exp(-x)) / 2

        # 参数与公式
        V0 = self.V0
        a = self.a[self.local_index]
        b = self.b[self.local_index]
        dc = self.dc[self.local_index]
        f0 = self.f0[self.local_index]


        
        # 计算公式
        dV1dtau = 2 * V0 / (a * (Tno-P)) * safe_exp(-state / a) * safe_cosh(Tt1o / (a * (Tno-P)))
        dV2dtau = 2 * V0 / (a * (Tno-P)) * safe_exp(-state / a) * safe_cosh(Tt2o / (a * (Tno-P)))
        dV1dsigma = -2 * V0 * Tt1o / (a * (Tno-P)**2) * safe_exp(-state / a) * safe_cosh(Tt1o / (a * (Tno-P)))
        dV2dsigma = -2 * V0 * Tt2o / (a * (Tno-P)**2) * safe_exp(-state / a) * safe_cosh(Tt2o / (a * (Tno-P)))
        dV1dstate = -2 * V0 / a * safe_exp(-state / a) * safe_sinh(Tt1o / (a * (Tno-P)))
        dV2dstate = -2 * V0 / a * safe_exp(-state / a) * safe_sinh(Tt2o / (a * (Tno-P)))
        dstatedt = b / dc * (V0 * safe_exp((f0 - state) / b) - slipv)
        
        
        
        dtau1dt=(-AdotV1+shear_loading-self.mu/(2.0*self.Cs)*(dV1dsigma*(dsigmadt-dPdt)+dV1dstate*dstatedt))/(1.0+self.mu/(2.0*self.Cs)*dV1dtau)
        dtau2dt=(-AdotV2+shear_loading-self.mu/(2.0*self.Cs)*(dV2dsigma*(dsigmadt-dPdt)+dV2dstate*dstatedt))/(1.0+self.mu/(2.0*self.Cs)*dV2dtau)

        return dstatedt,dsigmadt*1e-6,dtau1dt*1e-6,dtau2dt*1e-6



        #print(rank,self.Av1s.shape)
        


    def GetTtstress(self,Stress,eleVec):
        Tra=[]
        #print(self.eleVec.shape)
        for i in range(Stress.shape[1]):
            # ev11,ev12,ev13=eleVec[i,0],eleVec[i,1],eleVec[i,2]
            # ev21,ev22,ev23=eleVec[i,3],eleVec[i,4],eleVec[i,5]
            ev31,ev32,ev33=eleVec[i,6],eleVec[i,7],eleVec[i,8]
            
            Tr1=Stress[0,i]*ev31+Stress[3,i]*ev32+Stress[4,i]*ev33
            Tr2=Stress[3,i]*ev31+Stress[1,i]*ev32+Stress[5,i]*ev33
            Trn=Stress[4,i]*ev31+Stress[5,i]*ev32+Stress[2,i]*ev33

            Tra.append([Tr1,Tr2,Trn])
        Tra=np.array(Tra)
        return Tra
    
    def calc_corefunc_disloc(self):
        print(f'rank {rank} calc_corefunc_disloc...')
        # Av1s=np.zeros([len(self.xgV),len(self.elelstF)])
        # Av2s=np.zeros([len(self.xgV),len(self.elelstF)])
        # Bvs=np.zeros([len(self.xgV),len(self.elelstF)])
        # Av1d=np.zeros([len(self.xgV),len(self.elelstF)])
        # Av2d=np.zeros([len(self.xgV),len(self.elelstF)])
        # Bvd=np.zeros([len(self.xgV),len(self.elelstF)])
        Kfv1=[]
        Kfv2=[]
        xg=np.copy(self.xgV)
        F=np.array([1.0,0,0])
        SS,DS,TS=1.0,0,0
        
        for k in range(len(self.local_index)):
            i=self.local_index[k]
            P1=np.copy(self.nodelst[self.elelstF[i,0]-1])
            P2=np.copy(self.nodelst[self.elelstF[i,1]-1])
            P3=np.copy(self.nodelst[self.elelstF[i,2]-1])
            #print(f"Batch {i + 1}/{len(self.elelstF)} completed")
            stress, _=triangle_dislocation_integral_stress_Kfv_python(xg,P1,P2,P3,SS,DS,TS,self.mu,self.lambda_,False)
            Kfv1.append(stress)
            #Stress1=stress.transpose()
            #print(Stress1.shape)
            # Tra=self.GetTtstress(Stress,self.eleVecV)
            # #print(Stress.shape,Tra.shape)
            # Av1s[:,i]=Tra[:, 0]
            # Av2s[:,i]=Tra[:, 1]
            # Bvs[:,i]=Tra[:, 2]
        
        SS,DS,TS=0.0,1.0,0
        for k in range(len(self.local_index)):
            i=self.local_index[k]
            P1=np.copy(self.nodelst[self.elelstF[i,0]-1])
            P2=np.copy(self.nodelst[self.elelstF[i,1]-1])
            P3=np.copy(self.nodelst[self.elelstF[i,2]-1])
            #print(f"Batch {i + 1}/{len(self.elelstF)} completed")
            stress, _=triangle_dislocation_integral_stress_Kfv_python(xg,P1,P2,P3,SS,DS,TS,self.mu,self.lambda_,False)
            Kfv2.append(stress)
            # Tra=self.GetTtstress(Stress,self.eleVecV)
            # Av1d[:,i]=Tra[:, 0]
            # Av2d[:,i]=Tra[:, 1]
            # Bvd[:,i]=Tra[:, 2]
        Kfv1=np.array(Kfv1)
        Kfv2=np.array(Kfv2)
        return Kfv1,Kfv2
   
            

        
    def writeVTU(self, fname,init=False):
        if not fname.endswith(".vtu"):
            fname += ".vtu"

        # 1. 创建非结构化网格
        ugrid = vtk.vtkUnstructuredGrid()

        # 2. 点
        points = vtk.vtkPoints()
        for i in range(self.nodelst.shape[0]):
            points.InsertNextPoint(float(self.nodelst[i][0]),
                                float(self.nodelst[i][1]),
                                float(self.nodelst[i][2]))
        ugrid.SetPoints(points)

        # 3. 单元（三角形）
        for i in range(self.elelst.shape[0]):
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, int(self.elelst[i][0]-1))
            tri.GetPointIds().SetId(1, int(self.elelst[i][1]-1))
            tri.GetPointIds().SetId(2, int(self.elelst[i][2]-1))
            ugrid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())

        # 4. 写入 CellData
        def add_scalar(name, arr):
            data = vtk.vtkFloatArray()
            data.SetName(name)
            for v in arr:
                data.InsertNextValue(float(v))
            ugrid.GetCellData().AddArray(data)

        add_scalar("Normal_[MPa]", self.Tno)
        
        add_scalar("Shear_[MPa]", self.Tt)
        add_scalar("Shear_1[MPa]", self.Tt1o)
        add_scalar("Shear_2[MPa]", self.Tt2o)
        add_scalar("rake[Degree]", self.rake*180./np.pi)
        add_scalar("state", self.state)
        add_scalar("Slipv[m/s]", self.slipv)
        add_scalar("Slipv1[m/s]", self.slipv1)
        add_scalar("Slipv2[m/s]", self.slipv2)
        add_scalar("fric", self.fric)
        add_scalar("slip[m]", self.slip)
        add_scalar("slip1[m]", self.slip1)
        add_scalar("slip2[m]", self.slip2)
        add_scalar("Strike Tracation from viscosity[MPa]", self.VFdot1*1e-6)
        add_scalar("Dip Tracation from viscosity[MPa]", self.VFdot2*1e-6)
        add_scalar("Normal Tracation from viscosity[MPa]", self.VFdot3*1e-6)
        if(self.Ifdila==True):
            add_scalar("Pore_pressure[MPa]", self.P*1e-6)
            add_scalar("Porosity[Degree]", self.porosity)
        if(self.Ifthermal==True):
            add_scalar("Temperature[Degree]", self.Tempe)
        if(init==True):
            add_scalar("a", self.a)
            add_scalar("b", self.b)
            add_scalar("a-b", self.a - self.b)
            add_scalar("dc", self.dc)
            if(self.Ifdila==True or self.Ifthermal==True):
                add_scalar("shear zone width[m]", self.hs)
            add_scalar("slip_plate[m/s]", self.slipvC)

        # 5. 写文件（binary + zlib 压缩）
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(ugrid)
        writer.SetDataModeToBinary()      # 二进制
        writer.SetCompressorTypeToZLib()  # 压缩
        writer.Write()

    def output_visco(self):
        from pathlib import Path
        dir_path = Path('visout')
        dir_path.mkdir(parents=True, exist_ok=True)
        f=open('visout/visco_surf_%d.txt'%self.step,'w')
        for i in range(len(self.xgV)):
            f.write('%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g\n' % (
                    self.xgV[i,0], self.xgV[i,1], self.xgV[i,2],
                    self.sigmaV[i,0], self.sigmaV[i,1], self.sigmaV[i,2],
                    self.sigmaV[i,3], self.sigmaV[i,4],self.sigmaV[i,5],self.sigmaV_rate[i]))
        f.close()
        np.save('out_npy/Isurf%d'%self.step,self.Isurf) 
        np.save('out_npy/Ivol%d'%self.step,self.Ivol)

    
    def plot_stres(self,stres):
        nx, ny = 300, 300
        x=self.xgV[:,0]
        y=self.xgV[:,1]
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        X, Y = np.meshgrid(xi, yi)

        # 插值
        fig=plt.subplots(figsize=(10,6))
        Sxx = griddata((x,y), stres, (X,Y), method='nearest')
        im = plt.pcolormesh(X,Y,Sxx,
                       cmap='RdBu_r',
                       shading='auto')
        plt.colorbar()
        plt.savefig('1.png')
        plt.show()
