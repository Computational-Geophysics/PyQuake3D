import numpy as np
import struct
import matplotlib.pyplot as plt
from math import *
import os
import sys
#import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata
import pyquake3d.readmsh as readmsh
#import cupy as cp
from collections import deque
from scipy.ndimage import gaussian_filter1d
import pyquake3d.Hmatrix as Hmat
import joblib
from mpi4py import MPI
import pyquake3d.QDsim as QDsim
import pyvista as pv
from scipy.linalg import lu_factor, lu_solve
import logging
from datetime import datetime
import vtk
import gc  # Garbage collection
import psutil
import torch
import pyquake3d.Hmatvec_gpu as Hmatvec
import torch.distributed as dist
from pyquake3d.config import comm, rank, size

import ctypes
from pathlib import Path

def balance_gpu_assignment(process_demands, num_gpus):
    """
    process_demands:  [(rank_id, data_size), ...]
    num_gpus: int, GPU number available on the node
    """
    
    sorted_processes = sorted(process_demands, key=lambda x: x[1], reverse=True)
    
    # 2. Initialize the GPU and store (current total size, [list of rank elements]).
    gpu_buckets = [[0, []] for _ in range(num_gpus)]
    
    # 3. Greedy allocation
    for p_rank, p_size in sorted_processes:
        # Find the currently least occupied GPU.
        target_gpu = min(gpu_buckets, key=lambda x: x[0])
        
        # allocation
        target_gpu[0] += p_size
        target_gpu[1].append(p_rank)
    
    # 4. Convert to a mapping table {rank: gpu_id}
    assignment = {}
    for gpu_id, bucket in enumerate(gpu_buckets):
        for p_rank in bucket[1]:
            assignment[p_rank] = gpu_id
            
    return assignment, gpu_buckets

class QDsim_gpumpi(QDsim.QDsim):
    def __init__(self,elelstF,nodelst,Para):
        super().__init__(elelstF,nodelst,Para)
    

    

    def get_rank_Mat_mem(self):
        matmemo=0.0
        for i in range(len(self.local_blocks)):
            if(hasattr(self.local_blocks[i], 'judaca') and self.local_blocks[i].judaca==True):
                matmemo += (self.local_blocks[i].A.nbytes + self.local_blocks[i].B.nbytes)/(1024*1024)/1024.0
            else:
                matmemo += self.local_blocks[i].Mf_A1s.nbytes*6.0/(1024*1024)/1024.0
        demands = comm.gather((rank, matmemo), root=0)
        print('demands:',demands)
        rank_to_gpu_map = None
        if rank == 0:
            #num_gpus = torch.cuda.device_count() #
            num_gpus=self.GPU_cores
            print(f"Rank 0: get all memory cost,assign {num_gpus}  GPUs...")
            
            # 调用算法
            rank_to_gpu_map, buckets = balance_gpu_assignment(demands, num_gpus)
            for i, b in enumerate(buckets):
                print(f" >> GPU {i} predicted load: {b[0]:.2f} GiB | process: {b[1]}")
        rank_to_gpu_map = comm.bcast(rank_to_gpu_map, root=0)
        assigned_gpu_id = rank_to_gpu_map[rank]
        device = f'cuda:{assigned_gpu_id}'
        return device
    
    def init_torchtensor(self):
        self.GPU_cores=self.Para0['GPU_cores']
        #self.device='cpu'
        #self.device=f'cuda:{rank}'
        #cores_pergpu=int(size/self.GPU_cores)
        # grank=int(rank%self.GPU_cores)
        # self.device=f'cuda:{grank}' if torch.cuda.is_available() else 'cpu'
        # print('device type:',self.device)
        self.device='cpu'
        if torch.cuda.is_available():
            self.device=self.get_rank_Mat_mem()
        
        #torch.cuda.reset_peak_memory_stats()
        self.Hmatrix_tensor=Hmatvec.BatchedMatVecPreprocessor(self.device)
        self.Hmatrix_tensor.transfer_hmatrix(self.local_blocks,len(self.eleVec))
        
        print(f"rank {rank} turely assigned for single GPU: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GiB")
        print(f"rank {rank} PyTorch reserved for single GPU: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GiB")
        print(f"rank {rank} Peak assigned for single GPU: {torch.cuda.max_memory_allocated(self.device) / 1024**3:.2f} GiB")
    




    def simu_forward_mpi_tensor(self,dttry):
        slipv1=self.slipv1-self.slipvC*np.cos(self.rake0)
        slipv2=self.slipv2-self.slipvC*np.sin(self.rake0)
        #slipv_tensor=torch.from_numpy(self.slipv).to(self.device)
        t0 = MPI.Wtime()
        slip_tensor = torch.stack(
            (torch.from_numpy(slipv1), torch.from_numpy(slipv2)), dim=1
        ).to(self.device)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)
        #Calculating Kv first
        comm.Barrier()
        t0 = MPI.Wtime()

        
        traction_rate = self.Hmatrix_tensor.hmatrix_matvec_2d(slip_tensor).reshape(-1, 3)
        AdotV1 = traction_rate[:, 0]
        AdotV2 = traction_rate[:, 1]
        if self.fix_Tn:
            dsigmadt = self.normal_loading
        else:
            dsigmadt = traction_rate[:, 2]
        #print(AdotV1,AdotV2)
        
        
        t1 = MPI.Wtime()
        self.compute_time += (t1 - t0)
        
        #Combine results from all ranks
        t0 = MPI.Wtime()
        Ne=len(slipv1)
        #
        # self.dsigmadt=comm.allreduce(dsigmadt.cpu().numpy(), op=MPI.SUM)
        # self.AdotV1=comm.allreduce(AdotV1.cpu().numpy(), op=MPI.SUM)
        # self.AdotV2=comm.allreduce(AdotV2.cpu().numpy(), op=MPI.SUM)
        if(self.fix_Tn==True):
            self.dsigmadt=self.normal_loading
            buf = torch.cat([AdotV1, AdotV2]).cpu().numpy()
            #buf = np.concatenate([AdotV1.cpu().numpy(),AdotV2.cpu().numpy()])
            buf = comm.allreduce(buf, op=MPI.SUM)
            self.AdotV1=buf[:Ne]
            self.AdotV2=buf[Ne:]
        else:
            buf = torch.cat([dsigmadt, AdotV1, AdotV2]).cpu().numpy()
            #buf = np.concatenate([dsigmadt.cpu().numpy(),AdotV1.cpu().numpy(),AdotV2.cpu().numpy()])
            buf = comm.allreduce(buf, op=MPI.SUM)
            self.dsigmadt=buf[:Ne]
            self.AdotV1=buf[Ne:Ne*2]
            self.AdotV2=buf[Ne*2:]
        
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)


        
        nrjct=0
        h=dttry
        running=True
        dtnext=None

        while running:
            t0 = MPI.Wtime()
            Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk=self.RungeKutte_solve_Dormand_Prince_(h)
            t1 = MPI.Wtime()
            self.RK_time += (t1 - t0)
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

        
        indexmin=np.where(self.slipv<1e-30)[0]
        if(len(indexmin)>0):
            self.slipv[indexmin]=1e-30
        #self.maxslipv0=np.max(self.slipv)
        self.slip1=self.slip1+self.slipv1*h
        self.slip2=self.slip2+self.slipv2*h
        self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)
        

        if(self.step%self.Para0['outsteps']==0):
            #print(self.counts, self.displs,self.Tno.shape,Tno_yhk.shape)
            #print(Tno_yhk.dtype, self.Tno.dtype)
            t0 = MPI.Wtime()
            comm.Gatherv(sendbuf=Tno_yhk,recvbuf=(self.Tno, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=Tt1o_yhk,recvbuf=(self.Tt1o, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=Tt2o_yhk,recvbuf=(self.Tt2o, (self.counts, self.displs)), root=0)
            comm.Gatherv(sendbuf=state_yhk,recvbuf=(self.state, (self.counts, self.displs)), root=0)
            
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
                self.rake=np.arctan2(self.Tt2o,self.Tt1o)
                self.fric=self.Tt/(self.Tno-self.P*1e-6)
        
        return h,dtnext
    
    def simu_forward_mpi_LTM_tensor(self,dttry):
        
        cart_rank = self.cart_comm.Get_rank()
        row, col = self.cart_comm.Get_coords(cart_rank)
        slipv1=np.zeros(len(self.slipv1))
        slipv2=np.zeros(len(self.slipv2))
        slipv1[self.local_slipv_index]=self.slipv1[self.local_slipv_index]-self.slipvC[self.local_slipv_index]*np.cos(self.rake0[self.local_slipv_index])
        slipv2[self.local_slipv_index]=self.slipv2[self.local_slipv_index]-self.slipvC[self.local_slipv_index]*np.sin(self.rake0[self.local_slipv_index])
        # color = 0 if row == col else MPI.UNDEFINED  
        # diag_comm = comm.Split(color, key=cart_rank)
        #print('start:',self.step,self.Tno.shape,rank,cart_rank)
        t0 = MPI.Wtime()
        slip_tensor = torch.stack(
            (torch.from_numpy(slipv1), torch.from_numpy(slipv2)), dim=1
        ).to(self.device)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)
        
        t0 = MPI.Wtime()
        traction_rate = self.Hmatrix_tensor.hmatrix_matvec_2d(slip_tensor).reshape(-1, 3)
        AdotV1 = traction_rate[:, 0]
        AdotV2 = traction_rate[:, 1]
        if self.fix_Tn:
            dsigmadt = self.normal_loading
        else:
            dsigmadt = traction_rate[:, 2]

        t1 = MPI.Wtime()
        self.compute_time += (t1 - t0)


        #Combine results from all ranks
        t0 = MPI.Wtime()
        Ne=len(slipv1)
        if(self.fix_Tn==True):
            self.dsigmadt=self.normal_loading
            buf = torch.cat([AdotV1, AdotV2]).cpu().numpy()
            buf = comm.allreduce(buf, op=MPI.SUM)
            self.AdotV1=buf[:Ne]
            self.AdotV2=buf[Ne:]
        else:
            buf = torch.cat([dsigmadt, AdotV1, AdotV2]).cpu().numpy()
            buf = comm.allreduce(buf, op=MPI.SUM)
            self.dsigmadt=buf[:Ne]
            self.AdotV1=buf[Ne:Ne*2]
            self.AdotV2=buf[Ne*2:]

        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)

        
        #row_comm = self.cart_comm.Sub(remain_dims=[False, True])
        #print(row,col,cart_rank,row_comm.rank)
        diag_rank_in_row = 0  # In a row communicator, the local rank of a diagonal process is the row number.
        M=len(self.xg)
        # Receive buffer: All processes must provide this, but only root uses the result.

        sendbuf_v1 = np.array(self.AdotV1, dtype=np.float64)
        sendbuf_v2 = np.array(self.AdotV2, dtype=np.float64)
        sendbuf_sig = np.array(self.dsigmadt, dtype=np.float64)

        if col == 0:
            recvbuf_v1  = np.zeros(M, dtype=np.float64)
            recvbuf_v2  = np.zeros(M, dtype=np.float64)
            recvbuf_sig = np.zeros(M, dtype=np.float64)
        else:
            recvbuf_v1 = np.empty(M, dtype=np.float64)
            recvbuf_v2 = np.empty(M, dtype=np.float64)
            recvbuf_sig = np.empty(M, dtype=np.float64)
        #print(rank)
        t0 = MPI.Wtime()
        # row_comm Reduce
        self.row_comm.Reduce(sendbuf_v1,  recvbuf_v1,  op=MPI.SUM, root=diag_rank_in_row)
        self.row_comm.Reduce(sendbuf_v2,  recvbuf_v2,  op=MPI.SUM, root=diag_rank_in_row)
        self.row_comm.Reduce(sendbuf_sig, recvbuf_sig, op=MPI.SUM, root=diag_rank_in_row)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)
        if col == 0:
            self.AdotV1 = recvbuf_v1
            self.AdotV2 = recvbuf_v2
            self.dsigmadt = recvbuf_sig
            #print(np.max(self.AdotV1),rank)

            
        nrjct=0
        h=dttry
        running=True
        dtnext=None
        
        if(col == 0):
            
            while running:
                t0= MPI.Wtime()
                Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk=self.RungeKutte_solve_Dormand_Prince_(h)
                t1 = MPI.Wtime()
                self.RK_time += (t1 - t0)
                recvbuf_Relerror1 = np.zeros(1, dtype=np.float64) if cart_rank == 0 else np.empty(1, dtype=np.float64)
                recvbuf_Relerror1 = np.ascontiguousarray(recvbuf_Relerror1)
                self.diag_comm.Reduce(self.Relerrormax1, recvbuf_Relerror1, op=MPI.MAX, root=0)
                
                recvbuf_Relerror2 = np.zeros(1, dtype=np.float64) if cart_rank == 0 else np.empty(1, dtype=np.float64)
                recvbuf_Relerror2 = np.ascontiguousarray(recvbuf_Relerror2)
                self.diag_comm.Reduce(self.Relerrormax2, recvbuf_Relerror2, op=MPI.MAX, root=0)
                if(cart_rank==0):
                    self.RelTol1=1e-4
                    self.RelTol2=1e-4
                    condition1=recvbuf_Relerror1[0]/self.RelTol1
                    condition2=recvbuf_Relerror2[0]/self.RelTol2
                    hnew1=h*0.9*(self.RelTol1/recvbuf_Relerror1[0])**0.2
                    hnew2=h*0.9*(self.RelTol2/recvbuf_Relerror2[0])**0.2
                    dtnext_raw = min(hnew1, hnew2)
                    if(max(condition1,condition2)<1.0 and not (np.isnan(condition1) or np.isnan(condition2))):
                        #print(type(hnew1),type(condition1))
                        dtnext=min(1.5*h,dtnext_raw)
                        accept = 1.0
                    
                    
                    else:
                        nrjct=nrjct+1
                        dtnext = max(0.5 * h, dtnext_raw)  
                        h = dtnext
                        
                        #h=0.5*h
                        #print('nrjct:',nrjct,'  condition1,',condition1,' condition2:',condition2,'  dt:',h)
                        accept = 0.0
                        if(h<1.e-15 or nrjct>20):
                            print('error: dt is too small')
                            accept = -1.0

                else:
                    dtnext = 0.0
                    accept = 0.0
                bcast_data = np.array([dtnext, accept], dtype=np.float64)
                recv_bcast = np.zeros(2, dtype=np.float64)

                self.diag_comm.Bcast(bcast_data if cart_rank == 0 else recv_bcast, root=0)

                # 5. All diagonal processes parse broadcasts.
                if cart_rank != 0:
                    dtnext = recv_bcast[0]
                    accept = recv_bcast[1]

                # 6. All processes should determine when to exit.
                if accept > 0.5:        # accept == 1.0 → break
                    self.dtnext = dtnext
                    break
                elif accept < -0.5:     # accept == -1.0 → Error termination
                    if cart_rank == 0:
                        print("Simulation failed: dt too small.")
                    comm.Abort(1)  # stop all pro
                    break
                else:                   # accept == 0.0 → Rejection, continue the cycle
                    h = dtnext  # update h for the next trial
            
            self.time=self.time+h
            #if(rank==0):
            #update slip rate and rake
            Tno_yhk[Tno_yhk<0.1]=0.1
            self.Tno_local=Tno_yhk
            self.Tt1o_local=Tt1o_yhk
            self.Tt2o_local=Tt2o_yhk
            self.state_local=state_yhk
            
            #self.Tt_local=np.sqrt(Tt1o_yhk*Tt1o_yhk+Tt2o_yhk*Tt2o_yhk)
            #print('self.Tt1o',np.mean(self.Tt1o),np.mean(self.Tt2o))
            self.slipv1[:]=0
            self.slipv2[:]=0
            self.slipv1[self.local_index]=(2.0*self.V0)*np.exp(-self.state_local/self.a[self.local_index])*np.sinh(self.Tt1o_local/(self.Tno_local-self.P[self.local_index]*1e-6)/self.a[self.local_index])
            self.slipv2[self.local_index]=(2.0*self.V0)*np.exp(-self.state_local/self.a[self.local_index])*np.sinh(self.Tt2o_local/(self.Tno_local-self.P[self.local_index]*1e-6)/self.a[self.local_index])
            #print(np.max(np.sinh(self.Tt1o_local/self.Tno_local/self.a[self.local_index])),rank)
            #print(np.max(np.exp(-self.state_local/self.a[self.local_index])),rank,len(self.local_index))
            # slipv1_rec = np.zeros(len(self.slipv1), dtype=np.float64) if cart_rank == 0 else np.empty(1, dtype=np.float64)
            # diag_comm.Reduce(self.slipv1, slipv1_rec, op=MPI.SUM, root=0)
            t0 = MPI.Wtime()
            self.slipv1=self.diag_comm.allreduce(self.slipv1, op=MPI.SUM)
            self.slipv2=self.diag_comm.allreduce(self.slipv2, op=MPI.SUM)
            t1 = MPI.Wtime()
            self.comm_time += (t1 - t0)

            self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)
            
            #self.slipv=diag_comm.allreduce(self.slipv, op=MPI.SUM)
            indexmin=np.where(self.slipv<1e-30)[0]
            if(len(indexmin)>0):
                self.slipv[indexmin]=1e-30
            #self.maxslipv0=np.max(self.slipv)
            #print(self.Tno.shape,rank,cart_rank)
            #update slip
            self.slip1=self.slip1+self.slipv1*h
            self.slip2=self.slip2+self.slipv2*h
            self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)
            if(self.step%self.Para0['outsteps']==0):
                
                self.Tno[:]=0
                self.Tt1o[:]=0
                self.Tt2o[:]=0
                self.state[:]=0
                self.Tno[self.local_index]=Tno_yhk
                self.Tt1o[self.local_index]=Tt1o_yhk
                self.Tt2o[self.local_index]=Tt2o_yhk
                self.state[self.local_index]=state_yhk
            #     #print(self.counts, self.displs,self.Tno.shape,Tno_yhk.shape)
            #     #print(Tno_yhk.dtype, self.Tno.dtype)

                t0 = MPI.Wtime()
                recvbuf = np.zeros(len(self.Tno), dtype=np.float64)
                self.diag_comm.Reduce(self.Tno, recvbuf, op=MPI.SUM, root=0)
                if(cart_rank==0):
                    self.Tno=recvbuf

                recvbuf = np.zeros(len(self.Tt1o), dtype=np.float64) 
                self.diag_comm.Reduce(self.Tt1o, recvbuf, op=MPI.SUM, root=0)
                if(cart_rank==0):
                    self.Tt1o=recvbuf

                recvbuf = np.zeros(len(self.Tt2o), dtype=np.float64) 
                self.diag_comm.Reduce(self.Tt2o, recvbuf, op=MPI.SUM, root=0)
                if(cart_rank==0):
                    self.Tt2o=recvbuf

                recvbuf = np.zeros(len(self.state), dtype=np.float64) 
                self.diag_comm.Reduce(self.state, recvbuf, op=MPI.SUM, root=0)
                if(cart_rank==0):
                    self.state=recvbuf

                if(self.Ifthermal==True):
                    self.Tempe[self.index_]=0
                    recvbuf = np.zeros(len(self.Tempe), dtype=np.float64) 
                    self.diag_comm.Reduce(self.Tempe, recvbuf, op=MPI.SUM, root=0)
                    if(cart_rank==0):
                        self.Tempe=recvbuf

                if(self.Ifdila==True):
                    self.P[self.index_]=0
                    recvbuf = np.zeros(len(self.P), dtype=np.float64) 
                    self.diag_comm.Reduce(self.P, recvbuf, op=MPI.SUM, root=0)
                    if(cart_rank==0):
                        self.P=recvbuf
                    
                    self.porosity[self.index_]=0
                    recvbuf = np.zeros(len(self.porosity), dtype=np.float64) 
                    self.diag_comm.Reduce(self.porosity, recvbuf, op=MPI.SUM, root=0)
                    if(cart_rank==0):
                        self.porosity=recvbuf
                    
                
                t1 = MPI.Wtime()
                self.comm_time += (t1 - t0)

                if(cart_rank==0):
                    self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
                    self.rake=np.arctan2(self.Tt2o,self.Tt1o)
                    self.fric=self.Tt/(self.Tno-self.P*1e-6)
                
                


                

        #bcast slipv in row_comm
        t1 = MPI.Wtime()
        self.row_comm.Bcast(self.slipv1, root=diag_rank_in_row)
        self.row_comm.Bcast(self.slipv2, root=diag_rank_in_row)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)
        if(self.Ifthermal==True):
            if(col == 0):
                Tempe,dTdt0,Tarr=self.Calc_T_implicit_mpi(h)
                self.Tempe=Tempe
                self.dTdt0=dTdt0
                self.Tempearr=Tarr
        if(self.Ifdila==True):
            if(col == 0):
                Pre,dPdt0,Parr=self.Calc_P_implicit_mpi(h)
                self.dPdt0=dPdt0
                self.P=Pre
                self.Parr=Parr
        return h,dtnext


    def start_gpu(self):
        start_time = MPI.Wtime()
        SLIPV=[]
        Tt=[]
        #self.Init_mpi_local_variables()
        self.init_mpi_local_variables()
        
        totaloutputsteps=int(self.Para0['totaloutputsteps']) #total time steps
        file = open(self.state_file, "a", encoding="utf-8")
        self.monitor_total_memory(comm, prefix=f"Init:")
        file.write('iteration time_step(s) maximum_slip1_rate(m/s) maximum_slip2_rate(m/s) time(s) time(d)\n')
        
        for i in range(totaloutputsteps):
            self.step=i
            #print('!!!!!!!!!!!!!!',i)
            if(i==0):#inital step length
                dttry=self.htry
            else:
                dttry=dtnext
            
            if(self.Lt_jud==True):
                dttry,dtnext=self.simu_forward_mpi_LTM_tensor(dttry)
            else:
                dttry,dtnext=self.simu_forward_mpi_tensor(dttry)
            if(rank==0):
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


                    #SLIP=[]
                    SLIPV=[]
                    #Tt=[]
                    #output vtk
                    if(self.Para0['outputvtu']==True):
                        #print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                        fname=directory+'/step'+str(i)+'.vtu'
                        self.writeVTU(fname)
                    # if(self.Para0['outputmatrix']==True):
                    #     fname='step'+str(i)
                    #     self.writeVTU(fname)

    
    
        end_time = MPI.Wtime()
        #print(f"rank {rank} computation time: {self.compute_time:.6f} sec")
        if rank == 0:
            visible_gpus = torch.cuda.device_count()
            print(f"Visible GPUs: {visible_gpus}")

            print(f"Program run time: {end_time - start_time:.6f} sec")
            print(f"communication time: {self.comm_time:.6f} sec")
            print(f"Matrix product computation time: {self.compute_time:.6f} sec")
            print(f"Rugger-Kutta iteration time: {self.RK_time:.6f} sec")
            timetake=end_time - start_time
            file.write(f"Program run time: {end_time - start_time:.6f} sec")
            file.write(f"communication time: {self.comm_time:.6f} sec")
            file.write(f"Matrix product computation time: {self.compute_time:.6f} sec")
            file.write(f"Rugger-Kutta iteration time: {self.RK_time:.6f} sec")
            file.write('Program end time: %s\n'%str(datetime.now()))
            #file.write("Time taken: %.2f seconds\n"%timetake)
            file.close()
