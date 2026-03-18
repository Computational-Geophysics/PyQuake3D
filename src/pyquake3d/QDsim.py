import numpy as np
import struct
import matplotlib.pyplot as plt
from math import *
import pyquake3d.SH_greenfunction as SH_greenfunction
import pyquake3d.DH_greenfunction as DH_greenfunction
import os
import sys
#import json
from concurrent.futures import ProcessPoolExecutor
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata
import pyquake3d.readmsh as readmsh
#import cupy as cp
from collections import deque
from scipy.ndimage import gaussian_filter1d
import pyquake3d.Hmatrix as Hmat
import joblib
import pyquake3d.config as config
from pyquake3d.config import comm, rank, size
from mpi4py import MPI
import pyvista as pv
from scipy.linalg import lu_factor, lu_solve
import logging
from datetime import datetime
import vtk
import gc  # Garbage collection
import psutil





def get_sumS(X,Y,Z,nodelst,elelst):
    Ts,Ss,Ds=0,0,1
    mu=0.33e11
    lambda_=0.33e11
    Strs=[]
    Stra=[]
    Dis=[]
    for i in range(len(elelst)):
        P1=np.copy(nodelst[elelst[i,0]-1])
        P2=np.copy(nodelst[elelst[i,1]-1])
        P3=np.copy(nodelst[elelst[i,2]-1])
        Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda_)

        P1=np.copy(nodelst[elelst[i,0]-1])
        P2=np.copy(nodelst[elelst[i,1]-1])
        P3=np.copy(nodelst[elelst[i,2]-1])
        ue,un,uv=DH_greenfunction.TDdispHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,0.25)
        
        Dis_tems=np.array([ue,un,uv])
        #print(ue.shape,un.shape)
        if(len(Strs)==0):
            Strs=Stress
            Stra=Strain
            Dis=Dis_tems
        else:
            Strs=Strs+Stress
            Stra=Stra+Strain
            Dis=Dis+Dis_tems
    return Dis,Strs,Stra

# find the mesh boundary_edges and nodes
def find_boundary_edges_and_nodes(triangles):
    from collections import defaultdict
    edge_count = defaultdict(int)
    boundary_nodes = set()

    # 遍历每个三角形，统计边的出现次数
    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]
        for edge in edges:
            edge_count[edge] += 1

    # 找到只出现一次的边，并记录边上的节点
    boundary_edges = []
    for edge, count in edge_count.items():
        if count == 1:
            boundary_edges.append(edge)
            boundary_nodes.update(edge)

    return boundary_edges, np.array(list(boundary_nodes))

from scipy.spatial.distance import cdist

# Calculate the distance between two node coord
def find_min_euclidean_distance(coords1, coords2):
    # 使用 scipy.spatial.distance.cdist 计算成对距离
    distances = cdist(coords1, coords2, 'euclidean')
    # 找到最小距离及其对应的索引
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_idx]
    return min_distance


# Read parameters
def readPara0(fname):
    Para0={}
    f=open(fname,'r')
    for line in f:
        #print(line)
        if ':' in line:
            tem=line.split(':')
            #if(tem[0]==)
            Para0[tem[0].strip()]=tem[1].strip()
    return Para0

def clac_revn_vchange_filter(v):
    sigma = 2
    smoothed_y = gaussian_filter1d(v, sigma=sigma)
    revn=[]
    vchange=[]
    for i in range(len(v)):
        vchange.append(abs((v[i]-smoothed_y[i])/smoothed_y[i]))
    #plt.plot(v)
    #plt.plot(smoothed_y)
    #vchange=np.array(vchange)*(1.0-exp(-10000*np.mean(v)))
    if(np.mean(v)>0.001):
        vchange=np.array(vchange)*0.0
    #plt.show()
    return np.mean(vchange)




class QDsim:
    def __init__(self,elelst,nodelst,Para):
        #for i in range(len(xg)):
        
        fnamePara=Para['parameter directory']
        last_backslash_index = fnamePara.rfind('/')
        self.compute_time = 0.0
        self.comm_time = 0.0
        # get Parameter file name
        if last_backslash_index != -1:
            self.dirname = fnamePara[:last_backslash_index]
        else:
            self.dirname = fnamePara
        #print(self.dirname)
        self.Para0=Para
        #parameter define
        self.Corefunc_directory=self.Para0['Corefunc directory']
        self.save_corefunc=self.Para0['save Corefunc']
        jud_ele_order=self.Para0['Node_order']
        jud_scalekm=self.Para0['Scale_km']
        self.mu=self.Para0['Shear modulus']
        self.lambda_=self.Para0['Lame constants']
        self.density=self.Para0['Rock density']
        self.halfspace_jud=self.Para0['Half space']
        self.InputHetoparamter=self.Para0['InputHetoparamter']
        self.hmatrix_mpi_plot=self.Para0['Hmatrix_mpi_plot']
        self.Ifdila=False
        self.Ifthermal=False
        
        #self.halfwidth=self.Para0['Half width']

        try:
            
            self.Ifdila=self.Para0['If Dilatancy']
            self.Ifcouple=self.Para0['If Coupledthermal']
            self.DilatancyC=self.Para0['Dilatancy coefficient']
            self.Chyd=self.Para0['Hydraulic diffusivity']
            #self.hw=float(self.Para0['Low permeability zone thickness'])
            #self.hs=self.Para0['Actively shearing zone thickness']
            self.EPermeability=self.Para0['Effective compressibility']
        except:
            print('No Dilatancy parameters or incomplete parameters.')

        try:
            #thermal pressurizationparameter
            self.Ifthermal=self.Para0['If thermal']
            self.cth=self.Para0['Thermal diffusivity']
            self.At=self.Para0['Ratio of thermal expansivity to compressibility']
            self.c=self.Para0['Heat capacity']
            
        except:
            print('No thermal parameters or incomplete parameters.')
        
        #self.useGPU=self.Para0['GPU']=='True'
        self.useC=self.Para0['Using C++ green function']=='False'
        
        #self.tf=2.0*self.Chyd/self.hs/self.hw
        self.Lt_jud=False

        
        if(jud_scalekm==False):
            nodelst=nodelst/1e3
        #jud_ele_order=False
        # get element label and element center coodinate
        eleVec,xg=readmsh.get_eleVec(nodelst,elelst,jud_ele_order)
        self.eleVec=eleVec
        self.elelst=elelst
        self.nodelst=nodelst
        self.xg=np.array(xg, dtype=np.float64)
        
        self.maxslipvque=deque(maxlen=20)
        self.val_nv=0
        
        

        
        self.htry=1e-3
        self.Cs=sqrt(self.mu/self.density)
        self.time=0
        
        #self.useGPU=self.Para0['GPU']=='True'
        #self.num_process=int(self.Para0['Processors'])
        #self.Batch_size=int(self.Para0['Batch_size'])
        self.YoungsM=self.mu*(3.0*self.lambda_+2.0*self.mu)/(self.lambda_+self.mu)
        self.possonratio=self.lambda_/2.0/(self.lambda_+self.mu)
        
        # log_file = os.path.join('run_pyquake3d.log')
        # if os.path.exists(log_file):
        #     os.remove(log_file)
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     handlers=[
        #         logging.FileHandler(log_file, encoding='utf-8'),
        #         logging.StreamHandler()
        #     ]
        # )
        
        
        print('First Lamé constants',self.lambda_)
        print('Shear Modulus',self.mu)
        print('Youngs Modulus',self.YoungsM)
        print('Poissons ratio',self.possonratio)
        
        self.Init_condition()

        
        #self.calc_corefunc()
        #Calcultae Hmatrix structure
        #print('Calcultae Hmatrix structure..', flush=True)
        #self.tree_block=Hmat.createHmatrix(self.xg,self.nodelst,self.elelst,self.eleVec,self.mu,self.lambda_,self.halfspace_jud,plotHmatrix=self.hmatrix_mpi_plot)
        #self.tree_block=Hmat.createHmatrix(self.xg,self.nodelst,self.elelst,self.eleVec,self.Para0)

        print('Number of Node',nodelst.shape, flush=True)
        print('Number of Element',elelst.shape, flush=True)
        
        
        self.state_file='./state.txt'
        # 第一次打开：用 "w" 模式清空旧文件
        file = open(self.state_file, "w", encoding="utf-8")

        file.write('Program start time: %s\n'%str(datetime.now()))
        file.write('Input msh geometry file:%s\n'%self.dirname)
        file.write('Input parameter file:%s\n'%self.dirname)
        file.write('Number of Node:%d\n'%nodelst.shape[0])
        file.write('Number of Element:%d\n'%elelst.shape[0])
        file.write('Cs:%f\n'%self.Cs)
        file.write('First Lamé constants:%f\n'%self.lambda_)
        file.write('Shear Modulus:%f\n'%self.mu)
        
        file.write('Youngs Modulus:%f\n'%self.YoungsM)
        file.write('Poissons ratio:%f\n'%self.possonratio)
        file.write('maximum element size:%f\n'%self.maxsize)
        file.write('average elesize:%f\n'%self.ave_elesize)
        file.write('Critical nucleation size:%f\n'%self.hRA)
        file.write('Cohesive zone::%f\n'%self.A0)
        

        
    
    def monitor_total_memory(self,comm, prefix="Total Memory"):
        """
        Monitor memory usage across all processes and aggregate total usage to the root process.
        Only displays total memory (RSS and VMS) without local per-rank details.
        
        :param comm: MPI communicator (e.g., MPI.COMM_WORLD or self.cart_comm)
        :param prefix: Print prefix for logging
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Each process gets local memory usage (in MB)
        gc.collect()  # Force garbage collection for more accurate peak value
        process = psutil.Process(os.getpid())
        local_rss = process.memory_info().rss / 1024 / 1024  # Physical memory (RSS)
        local_vms = process.memory_info().vms / 1024 / 1024  # Virtual memory (VMS)
        
        # MPI reduction: Sum local values to get total
        total_rss = comm.reduce(local_rss, op=MPI.SUM, root=0)
        total_vms = comm.reduce(local_vms, op=MPI.SUM, root=0)
        
        # Root process prints only aggregated totals
        if rank == 0:
            print(f"{prefix} | Total RSS (all {size} processes): {total_rss:.2f} MB")
            print(f"{prefix} | Total VMS (all {size} processes): {total_vms:.2f} MB")
            print(f"{prefix} | Average per process: RSS {total_rss / size:.2f} MB | VMS {total_vms / size:.2f} MB")
            file = open(self.state_file, "a", encoding="utf-8")
            file.write(f"{prefix} | Total RSS (all {size} processes): {total_rss:.2f} MB\n")
            file.write(f"{prefix} | Total VMS (all {size} processes): {total_vms:.2f} MB\n")
            file.write(f"{prefix} | Average per process: RSS {total_rss / size:.2f} MB | VMS {total_vms / size:.2f} MB\n")
            file.flush()

        return total_rss, total_vms  # Return totals (meaningful only on root; None on others)

    def collect_blocks(self,block,rankrow=0,rankcol=0):
        """ Recursively traverse all leaf nodes and collect blocks that need to calculate ACA """
        blocks_to_process = []
        def traverse(block):
            if block.is_leaf():
                block.rank_row=rankrow
                block.rank_col=rankcol
                blocks_to_process.append(block)
            else:
                for child in block.children:
                    traverse(child)
        traverse(block)
        return blocks_to_process

    def deploy_Hmatrix(self,HMobj):
        self.Lt_jud=HMobj.Lt_jud
        if(HMobj.Lt_jud==False):
            self.tree_block=HMobj.tree_block
        else:
            self.dims=HMobj.dims
            # dimsq=int(sqrt(size))
            # self.dims=[dimsq,dimsq]
            periods = [False, False]
            reorder = False
            config.init_cart(self.dims,periods=periods, reorder=reorder)
            self.cart_comm = config.get_cart()
            self.row_comm = self.cart_comm.Sub(remain_dims=[False, True])
            cart_rank = self.cart_comm.Get_rank()
            row, col = self.cart_comm.Get_coords(cart_rank)
            color = 0 if col == 0 else MPI.UNDEFINED # diag_comm color=0, others UNDEFINED
            self.diag_comm = comm.Split(color, key=cart_rank)
            # print(row, col,cart_rank)
            # if(col==0):
            #     print(row, col,cart_rank, self.diag_comm.Get_rank())
            # #self.tree_block=HMobj.LTMtree_block_lst[cart_rank]
            self.tree_block=HMobj.tree_block
            self.root_block=HMobj.root_block
            
            self.LTMtree_block_lst = HMobj.LTMtree_block_lst
            self.cart_coords=HMobj.cart_coords
            LMblocks_to_process=self.collect_blocks(HMobj.root_block)

            for i in range(len(LMblocks_to_process)):
                rankrow=self.cart_coords[i,0]
                rankcol=self.cart_coords[i,1]
                if(row==rankrow):
                    self.local_index=LMblocks_to_process[i].row_cluster
                    break

            #self.local_index=HMobj.root_block.row_cluster
            #print('self.local_index',len(self.local_index))
            #self.local_slipv_index=HMobj.root_block.col_cluster

            # index_target_rank=[]
            # Nd=self.dims[0]
            # NI=len(self.root_block.row_index)
            # LMblocks_to_process=self.collect_blocks(self.root_block)
            # Coord=[]
            #cart_rank=self.cart_comm.Get_rank()
            # for i in range(len(LMblocks_to_process)):
            #     rowI=round((LMblocks_to_process[i].row_index[-1]+1)/NI*Nd)-1
            #     colI=round((LMblocks_to_process[i].col_index[-1]+1)/NI*Nd)-1
            #     #Coord.append([rowI, colI])
            #     #print(rowI,colI)
            #     #if(rowI == 0 and colI == 0):
            #     target_coords = [rowI, colI]
            #     print(target_coords,cart_rank)
                # target_rank = self.cart_comm.Get_cart_rank(target_coords)
                # #index_target_rank.append(target_rank)

                # if(cart_rank==target_rank):
                #     self.local_index=LMblocks_to_process[i].row_cluster
            #         self.local_slipv_index=LMblocks_to_process[i].col_cluster
                
            # #print(index_target_rank)
            # self.LTMtree_block_lst = [HMobj.LTMtree_block_lst[i] for i in index_target_rank]

            

            # row, col = self.cart_comm.Get_coords(rank)
            #if(row==col):
            #np.save('local_index%d'%rank,self.local_index)
            
            # cart_comm = config.get_cart()
            # row, col = cart_comm.Get_coords(rank)
            # print(row,rank,self.local_index[:10])
            # if(row==dimsq-col-1):
                # print(row,col,rank,'!!!!!!!!!!!!')
                # print(len(HMobj.root_block.row_cluster),len(HMobj.root_block.col_cluster))
                #self.local_index=HMobj.root_block.row_cluster
        #print(type(self.tree_block))
        #self.HMCoord=HMobj.Coord


    def writestate(self, msg: str):
        # 之后每次写入都用追加模式
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def start_LMT(self):
        start_time = MPI.Wtime()
        totaloutputsteps=int(self.Para0['totaloutputsteps']) #total time steps
        #file = open(self.state_file, "a", encoding="utf-8")
        SLIPV=[]
        Tt=[]
        self.init_mpi_local_variables()
        # #print(rank)
        for i in range(totaloutputsteps):
        #for i in range(0):
            self.step=i
            if(i==0):#inital step length
                dttry=self.htry
            else:
                dttry=dtnext
            #dttry,dtnext=self.simu_forward_mpi_(dttry) #Forward modeling
            
            dttry,dtnext=self.simu_forward_mpi_LTM(dttry)
            if(rank==0):
                year=self.time/3600/24/365
                #if(i%10==0):
                print('iteration:',i, flush=True)
                print('dt:',dttry,' max_vel:',np.max(np.abs(self.slipv)),' min_vel:',np.min(np.abs(self.slipv)),' Porepressure max:',np.max(self.P),' Porepressure min:',np.min(self.P),' dpdt_max:',np.max((self.dPdt0)),' dpdt_min:',np.min((self.dPdt0)),' Seconds:',self.time,'  Days:',self.time/3600/24,
                'year',year, flush=True)

                if(self.Para0['outputvtu']==True):
                    outsteps=int(self.Para0['outsteps'])
                    directory='out_vtk'
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                    #output slipv and Tt
                    if(i%outsteps==0):
                        #print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                        fname=directory+'/step'+str(i)+'.vtu'
                        self.writeVTU(fname)
        
        end_time = MPI.Wtime()

        if rank == 0:
            
            print(f"Program run time: {end_time - start_time:.6f} sec")
            timetake=end_time - start_time
            # file.write('Program end time: %s\n'%str(datetime.now()))
            # file.write("Time taken: %.2f seconds\n"%timetake)
            # file.close()

    def countblock(self):
        s=0
        for i in range(len(self.local_blocks)):
            s=s+len(self.local_blocks[i].row_cluster)
        return s

    def start(self):
        start_time = MPI.Wtime()
        totaloutputsteps=int(self.Para0['totaloutputsteps']) #total time steps
        file = open(self.state_file, "a", encoding="utf-8")
        SLIPV=[]
        Tt=[]
        #self.Init_mpi_local_variables()
        self.init_mpi_local_variables()
        self.monitor_total_memory(comm, prefix=f"Init:")
        file.write('iteration time_step(s) maximum_slip1_rate(m/s) maximum_slip2_rate(m/s) time(s) time(h)\n')
        for i in range(totaloutputsteps):
        #for i in range(0):
            self.step=i
            if(i==0):#inital step length
                dttry=self.htry
            else:
                dttry=dtnext
            #dttry,dtnext=self.simu_forward_mpi_(dttry) #Forward modeling
            if(self.Lt_jud==False):
                dttry,dtnext=self.simu_forward_mpi_(dttry)
            else:
                dttry,dtnext=self.simu_forward_mpi_LTM(dttry)
            #sim0.simu_forward(dttry)
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

    
    #Determine whether Hmatrix has been calculated. If it has been calculated, read it directly
    def get_block_core(self):
        jud_coredir=True
        #directory = 'surface_core'
        
        if os.path.exists(self.Corefunc_directory):
            file_path = os.path.join(self.Corefunc_directory, 'blocks_to_process.joblib')
            if not os.path.exists(file_path):
                jud_coredir=False
        else:
            os.mkdir(self.Corefunc_directory)
            jud_coredir=False
        blocks_to_process=[]
        if(jud_coredir==True):
            blocks_to_process = joblib.load(self.Corefunc_directory+'/blocks_to_process.joblib')
        #make sure all submatices are loaded
        self.blocks_to_process=blocks_to_process
        for i in range(len(blocks_to_process)):
            if(hasattr(blocks_to_process[i], 'judproc') and blocks_to_process[i].judproc==False):
                jud_coredir=False
                break
        return jud_coredir,blocks_to_process
        # elif():
        #     self.tree_block.parallel_traverse_SVD(comm, rank, size)

    def get_block_LTMcore(self):
        jud_coredir=True
        #directory = 'surface_core'
        dir_exists = False
        cart_rank = self.cart_comm.Get_rank()
        if cart_rank == 0:
            if not os.path.exists(self.Corefunc_directory):
                os.mkdir(self.Corefunc_directory)   
                jud_coredir=False
            dir_exists = True 
        
        dir_exists = comm.bcast(dir_exists, root=0)
        comm.Barrier()
        
        if(dir_exists==True):
            file_path = os.path.join(self.Corefunc_directory, 'blocks_to_process%d.joblib'%cart_rank)
            if not os.path.exists(file_path):
                jud_coredir=False
        
        if(jud_coredir==True):
            blocks_to_process = joblib.load(self.Corefunc_directory+'/blocks_to_process%d.joblib'%cart_rank)
            #make sure all submatices are loaded
            self.local_blocks=blocks_to_process
            #print(type(self.local_blocks),len(self.local_blocks))
            for i in range(len(blocks_to_process)):
                if(hasattr(blocks_to_process[i], 'judproc') and blocks_to_process[i].judproc==False):
                    jud_coredir=False
                    break
        return jud_coredir
        # sendbuf = np.array(jud_coredir, dtype='?')  # '?' = MPI_BOOL
        # #  root receive
        # if rank == 0:
        #     recvbuf_jud_coredir = np.empty(size, dtype='?')  # 接收 size 个 bool
        # else:
        #     recvbuf_jud_coredir = None

        # #  Gather
        # comm.Gather(sendbuf, recvbuf_jud_coredir, root=0)

        # # root 
        # if rank == 0:
        #     #result_list = recvbuf_jud_coredir.tolist()  # 转为 Python list[bool]
        #     has_false = np.any(~recvbuf_jud_coredir)
            

    def parallel_cells_scatter_send(self,nsize=size):
        N=len(self.eleVec)
        index0=np.arange(0,N,1)
        local_index = None
        if rank == 0:
            print('Assign cells for rank calculation:', N)
    
            # Manually distribute tasks evenly
            counts = [N // nsize] * nsize
            for i in range(N % nsize):
                counts[i] += 1
            task_chunks = []
            start = 0
            for c in counts:
                task_chunks.append(index0[start:start+c])
                start += c
            for i in range(1, nsize):
                comm.send(task_chunks[i], dest=i, tag=77)
            local_index = task_chunks[0] 
        else:
            #Non-zero process receiving tasks
            
            local_index = comm.recv(source=0, tag=77)
        #print('rank',rank,' cells for local rank calculation',len(local_index))
        return local_index

    

    def split_blocks_equally(self,blocks_to_process_row, n_splits):
        """
        Split the blocks_to_process_row into n_splits parts, making the total length of row_cluster
        in each part as balanced as possible.
        
        Parameters:
        - blocks_to_process_row: list of objects, each object has a row_cluster attribute (list)
        - n_splits: int, the number of splits (>0, and <= len(blocks_to_process_row))
        
        Returns:
        - list of lists: the split sub-lists, preserving the original order
        """
        if n_splits <= 0 or n_splits > len(blocks_to_process_row):
            raise ValueError("n_splits must be between 1 and len(blocks_to_process_row)")
        
        # Calculate the weight for each block (length of row_cluster)
        weights = np.array([1.0+log10(len(block.row_cluster)*len(block.col_cluster)) for block in blocks_to_process_row])
        #weights = np.array([1.0 for block in blocks_to_process_row])
        #weights = np.array([1.0+exp(-(len(block.row_cluster)*len(block.col_cluster))*0.01) for block in blocks_to_process_row])
        
        if len(weights) == 0:
            return [[] for _ in range(n_splits)]
        
        # Compute cumulative weights
        cum_weights = np.cumsum(weights)
        total_weight = cum_weights[-1]
        
        # Ideal cut points (cumulative weight targets)
        ideal_cuts = np.array([i * total_weight / n_splits for i in range(1, n_splits)])
        
        # Find the closest actual index for each ideal cut point (using binary search-like minimization)
        splits = [0]  # Starting point
        for ideal in ideal_cuts:
            # Find the index in cum_weights closest to the ideal
            idx = np.argmin(np.abs(cum_weights - ideal))
            splits.append(idx + 1)  # +1 because the next segment starts after the cut
        
        splits.append(len(blocks_to_process_row))  # End point
        
        # Generate sub-lists
        sub_lists = []
        for i in range(n_splits):
            start, end = splits[i], splits[i+1]
            sub_lists.append(blocks_to_process_row[start:end])
        
        # Verification: Print total weights for each split (optional, for debugging)
        sub_weights = [sum(pow(len(b.row_cluster)*len(b.col_cluster),0.1) for b in sub) for sub in sub_lists]
        print(f"Split weights: {sub_weights} (total: {sum(sub_weights)}, ideal per split: {total_weight / n_splits:.2f})")
        
        return sub_lists
    
    
    def parallel_LTMblock_assign(self, blocks_to_process):
        rank = self.cart_comm.Get_rank()
        #size = self.cart_comm.Get_size()
        target_coords = self.cart_comm.Get_coords(rank)
        blocks_to_process_row = [[ ] for _ in range(self.dims[0])]
        blocks_to_process_row_all=[]
        if(rank==0):
            for i in range(len(blocks_to_process)):
                Irow=blocks_to_process[i].rank_row
                blocks_to_process_row[Irow].append(blocks_to_process[i])
                #for j in range()
            n_splits=self.dims[1]
            
            for i in range(len(blocks_to_process_row)):
                sub_lists=self.split_blocks_equally(blocks_to_process_row[i], n_splits)
                #print('sub_lists',[len(blocks) for blocks in sub_lists])
                blocks_to_process_row_all.append(sub_lists)

        local_blocks=self.tree_block.parallel_block_scatter_send_(blocks_to_process_row_all,self.dims[0],self.dims[1], plotHmatrix=self.hmatrix_mpi_plot)
        #print('self.local_blocks:',rank,len(self.local_blocks),self.local_blocks[0])
        local_slipv_index=[]
        #local_index=[]
        for i in range(len(local_blocks)):
            index_cols=local_blocks[i].col_cluster
            #index_rows=local_blocks[i].row_cluster
            local_slipv_index.append(index_cols)
            #local_index.append(index_rows)
        flattened_indices = np.concatenate(local_slipv_index)
        self.local_slipv_index = np.sort(np.unique(flattened_indices))
        #flattened_indices = np.concatenate(local_index)
        #self.local_index = np.sort(np.unique(flattened_indices))
        
        #if(target_coords[1]==0):
        #    print(len(self.local_index),len(self.local_slipv_index),rank)
        return local_blocks

    def get_LMTblocks(self):
        blocks_to_process=[]
        for i in range(len(self.LTMtree_block_lst)):
            blocks_=self.collect_blocks(self.LTMtree_block_lst[i].root_block,rankrow=self.cart_coords[i,0],rankcol=self.cart_coords[i,1])
            blocks_to_process.append(blocks_)
        return [item for sublist in blocks_to_process for item in sublist]
    #calculate greenfuncs accerlated by Hmatrix and MPI
    def calc_greenfuncs_mpi(self):
        # if(self.Lt_jud==True):
        #     rank = self.cart_comm.Get_rank()
        # bcast parameters to all ranks
        jud_coredir=None
        blocks_to_process=[]
        if(rank==0):
            #Determine whether Hmatrix has been calculated. If it has been calculated, read it directly
            jud_coredir,blocks_to_process=self.get_block_core()
            print('jud_coredir',jud_coredir) #if saved corefunc
            if(jud_coredir==False):
                print('Start to calculate Hmatrix...')
            else:
                print('Hmatrix reading...')
                        
            #test green functions
            # x=np.ones(len(self.elelst))
            # start_time = time.time()
            # for i in range(1):
            #     y=self.tree_block.blocks_process_MVM(x,blocks_to_process,'A2d')
            #     #print(y[:20])
            #     #print(np.max(y))
            # end_time = time.time()
            # print(f"Green func calc_MVM_fromC Time taken: {end_time - start_time:.10f} seconds")

            #calculate memorary
            s=0 
            for i in range(len(blocks_to_process)):
                if(blocks_to_process[i].judaca==True):
                    s1=blocks_to_process[i].ACA_dictS['U_ACA_A1s'].nbytes/(1024*1024)
                    s2=blocks_to_process[i].ACA_dictS['V_ACA_A1s'].nbytes/(1024*1024)
                    s=s+s1+s2
                else:
                    s=s+blocks_to_process[i].Mf_A1s.nbytes/(1024*1024)
            print('memorary:',s)
        
        jud_coredir = comm.bcast(jud_coredir, root=0)
        
        if(jud_coredir==False):#Calculate green functions and compress in Hmatrix
            #sim0.local_blocks=sim0.tree_block.parallel_traverse_SVD(sim0.Para0['Corefunc directory'],plotHmatrix=sim0.Para0['Hmatrix_mpi_plot'])
            
            if(rank==0):
                #Assign tasks for calculating green functions
                if(self.Lt_jud==True):
                    print('Assign tasks for lattice Hmatrix')
                    blocks_to_process=self.get_LMTblocks()
                self.tree_block.master(self.Para0['Corefunc directory'],blocks_to_process,size-1,save_corefunc=self.save_corefunc)
            else:
                #Calculat green functions
                self.tree_block.worker()
                #sim0.tree_block.master_scatter(sim0.Para0['Corefunc directory'],blocks_to_process,size)
                '''Assign forward modelling missions for each rank with completed blocks submatrice'''
            if(self.Lt_jud==False):
                self.local_blocks=self.tree_block.parallel_block_scatter_send(self.tree_block.blocks_to_process,plotHmatrix=self.Para0['Hmatrix_mpi_plot'])
            else:
                self.local_blocks=self.parallel_LTMblock_assign(self.tree_block.blocks_to_process)
        else:
            '''Assign forward modelling missions for each rank with completed blocks submatrice'''
            if(self.Lt_jud==False):
                
                self.local_blocks=self.tree_block.parallel_block_scatter_send(blocks_to_process,plotHmatrix=self.Para0['Hmatrix_mpi_plot'])
            else:
                self.local_blocks=self.parallel_LTMblock_assign(blocks_to_process)
       
        #if(self.Ifdila==True):
        #print('parallel_cells_scatter_send')
        
        if(self.Lt_jud==False):
            self.local_index=self.parallel_cells_scatter_send(nsize=size)
            #print(rank,self.local_index)


    def get_rotation1(self,x1):
        #print('x',x)
        xmin=0
        x=x1/1000
        if(x<70):
            theta=-65
        elif(x>=70.0 and x<85.0):
            # temx=((x-60.0)/10.0-1.0)*np.pi/2
            # theta=0.0-(sin(temx)+1.0)*12.5
            temx = ((x - 70.0) / 7.5 - 1.0)*np.pi / 2
            theta = -65- (sin(temx) + 1.0)*10.0
        else:
            theta=-85.0
        return theta

    

    def randompatch(self):
        xmin, xmax = -22, 22
        ymin, ymax = -28, -18

        # 生成 N 个随机点
        N = 25
        np.random.seed(42)
        x_random = np.random.uniform(xmin, xmax, N)*1e3
        y_random = np.random.uniform(ymin, ymax, N)*1e3
        sizeR_random=np.random.uniform(0.7, 2.0, N)*1e3

        for i in range(N):
            nuclearloc=[x_random[i],0,y_random[i]]
            distem=np.linalg.norm(self.xg-nuclearloc,axis=1)
            index1=np.where(distem<sizeR_random[i])[0]
            #print(len(index1),sizeR_random[i])
            self.a[index1]=0.01
            self.b[index1]=0.025
            self.dc[index1]=0.015
            #print(nuclearloc)
    
    #calc_nucleaszie and cohesivezone
    def calc_nucleaszie_cohesivezone(self):
        maxsize=0
        elesize=[]
        for i in range(len(self.eleVec)):
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])
            sizeA=np.linalg.norm(P1-P2)
            sizeB=np.linalg.norm(P1-P3)
            sizeC=np.linalg.norm(P2-P3)
            size0=np.max([sizeA,sizeB,sizeC])
            if(size0>maxsize):
                maxsize=size0
            elesize.append(size0)
        elesize=np.array(elesize)
        self.maxsize=maxsize
        self.ave_elesize=np.mean(elesize)
        b=np.max(self.b)
        a=np.min(self.a)
        #b=0.024
        #a=0.0185
        #b=0.025
        #a=0.01
        sigma=np.mean(self.Tno*1e6)
        #print(self.Tno)
        L=np.min(self.dc)
        #L=0.015
        #print('L:',L)
        print('a,b,L:',a,b,L)
        self.hRA=2.0/np.pi*self.mu*b*L/(b-a)/(b-a)/sigma
        self.hRA=self.hRA*np.pi*np.pi/4.0
        self.A0=9.0*np.pi/32*self.mu*L/(b*sigma)
        
        print('maximum element size',maxsize, flush=True)
        print('average elesize',self.ave_elesize, flush=True)
        print('Critical nucleation size',self.hRA, flush=True)
        print('Cohesive zone:',self.A0, flush=True)
        return maxsize
    
    #set heterogeneous plate slip rate
    def Grad_slpv_con(self,const):
        self.slipvC=np.ones(len(self.xg))*self.Vpl_con
        Vpl_min=1e-16
        if(const==False):
            for i in range(len(self.xg)):
                if(self.xg[i,2]<-5000):
                    self.slipvC[i]=self.Vpl_con
                else:
                    self.slipvC[i]=Vpl_min+abs(self.xg[i,2])/5000.0*(self.Vpl_con-Vpl_min)

    #set heterogeneous normal stress
    def Tn_edge(self):
        np.random.seed(42) 
        #self.Tno[i]=self.Tno[i]*exp(dis1)
        boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(self.elelst)
        boundary_coord=self.nodelst[boundary_nodes-1]
        index_surface=np.where(np.abs(boundary_coord[:,2]-0.0)<1e-5)[0]
        index_b=np.arange(0,len(boundary_coord),1)
        index_sb=np.setdiff1d(index_b,index_surface)
        boundary_coord_sb=boundary_coord[index_sb]
        Wedge=0.1*2.0*1e-2
        edTno=self.Tno[0]*0.7
        maxTno=self.Tno[0]
        Tno1=np.copy(self.Tno)
        index1=[]
        index0=[]
        for i in range(len(self.xg)):
            coords1=np.array([self.xg[i]])
            #print(coords1.shape, boundary_coord.shape)
            distem=find_min_euclidean_distance(coords1, boundary_coord_sb)
            dis=distem/Wedge
            
            if(dis<1.0):
                #self.a[i]=aVs-(aVs-aVw)*dis1
                self.Tno[i]=edTno-(edTno-maxTno)*dis
            else:
                index0.append(i)
                
        #print('index1',len(index1))
        index1=np.arange(0,len(self.Tno),1)
        Ns=len(self.Tno)
        sample = np.random.choice(index1, size=Ns, replace=False)
        #arr = np.random.normal(loc=0.0, scale=0.3, size=100)
        arr = np.random.uniform(low=-0.5, high=0.5, size=Ns)
        for i in range(Ns):
            j=sample[i]
            Tno1[j]=self.Tno[j]+self.Tno[j]*arr[i]
        
        Xmin=np.min(self.xg[:,0])
        Xmax=np.max(self.xg[:,0])
        Y=self.xg[:,1]
        #y1=np.max(self.xg[:,1])-self.xg[:,1]
        #Y=np.stack((y1, self.xg[:,2]), axis=1) 
        #Y = np.linalg.norm(Y, axis=1)
        Ymin=np.min(Y)
        Ymax=np.max(Y)
        xi = np.linspace(Xmin, Xmax, 100)
        yi = np.linspace(Ymin, Ymax, 100)
        #print(Xmin, Xmax,Ymin, Ymax)
        Xi, Yi = np.meshgrid(xi, yi)
        

        # 插值结果是一个二维数组
        Si = griddata((self.xg[:,0], self.xg[:,1]), Tno1, (Xi, Yi), method='nearest')  # <<< 这就是 Si

        # 之后可用于滤波
        from scipy.ndimage import gaussian_filter
        smooth_full = gaussian_filter(Si, sigma=4.0)
        #print('smooth_full',np.max(smooth_full))
        # plt.pcolor(smooth_full)
        # plt.savefig('1.png')
        # plt.show()
        
        #pt = np.array([[self.xg[:,0], self.xg[:,1]]])
        points=np.stack((Xi.flatten(), Yi.flatten()), axis=1) 
        #print(points.flatten())
        val = griddata(points, smooth_full.flatten(), self.xg[:,[0,1]], method='nearest')
        nan_indices = np.where(np.isnan(val))[0]
        val[nan_indices]=self.Tno[nan_indices]
        self.Tno[index0]=val[index0]
        
        #print('self.Tno',np.max(self.Tno),self.xg.shape,self.Tno)

        
        
    def getpatchs(self):
        Pcell=[26585,26933,34668,26296,26410,18719]
        #Pcell=[86595,169419,135384,110925,142049,178840]
        #Pcell=[64274,65260,72540,103871,87141,92757]
        xg0=self.xg[Pcell]
        np.random.seed(42)
        xmin, xmax=1.5,3
        N=6
        R_random = np.random.uniform(xmin, xmax, N)*1e3
        for i in range(len(self.xg)):
            #tem=min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge
            coords1=np.array([self.xg[i]])
            #distem=find_min_euclidean_distance(coords1, boundary_coord_sb)
            #print(coords1.shape, boundary_coord.shape)
            ratio_arr=[]
            for j in range(N):
                diff = coords1 - xg0[j]  # Broadcasting: subtracts xg0 from each row
                distances = np.sqrt(np.sum(diff**2, axis=1))
                ratio=distances/R_random[j]
                ratio_arr.append(ratio)
            # distances = np.sqrt(np.sum((coords1 - xg0)**2, axis=1))
            # min_distance = distances.min()
            minratio=np.min(ratio_arr)

            if(minratio<1.0):
                #print(i)
                self.a[i]=0.01
                self.b[i]=0.035
                #self.slipv[i]=self.Vpl_con
                self.dc[i]=0.035


    #set initial condition
    def Init_condition(self):
        N=len(self.eleVec)
        self.Relerrormax1_last=0
        self.Relerrormax2_last=0
        self.Tt1o=np.zeros(N)
        self.Tt2o=np.zeros(N)
        self.Tno=np.zeros(N)
        self.fix_Tn=self.Para0['Fix_Tn']
        ssv_scale=self.Para0['ssv_scale']
        ssh1_scale=self.Para0['ssh1_scale']
        ssh2_scale=self.Para0['ssh2_scale']
        trac_nor=self.Para0['Vertical principal stress value']
        
        
        
        
        self.P0=0
        self.P=np.zeros(N)
        self.dPdt0=np.zeros(N)
        
        self.local_index=np.arange(0,N,1)
        
        c=1e-2
        scaleC=0.001
        num=60
        ymax=100
        if(self.Ifdila==True):
            
            self.hs=self.Para0['Shearing zone width']*np.ones(N)
            self.P0=self.Para0['Constant porepressure']
            self.P=np.ones(N)*self.Para0['Initial porepressure']
            # self.yp=np.logspace(start=-3, stop=10, num=num, base=10)
            # self.zp=np.log(1+self.yp/c)
            # self.zp=np.linspace(log(c),log(30),num)
            # self.yp=-c+np.exp(self.zp)
            self.yp=np.logspace(start=log(scaleC*c), stop=log(ymax), num=num, base=np.e)-scaleC*c
            self.zp=np.log(c+self.yp)
            self.Parr=np.ones([N,num])*self.Para0['Initial porepressure']
            self.Pmatrix=np.zeros([num,num])
            self.Parr=self.Parr*1e6
            self.P0=self.P0*1e6
            self.P=self.P*1e6
            self.dTdtmatrix=np.zeros([N,num])
            self.Calc_Pmatrix()
            self.porosity=np.zeros(N)
            
        
        if(self.Ifthermal==True):
            self.hs=self.Para0['Shearing zone width']*np.ones(N)
            self.dTdt0=np.zeros(N)
            self.T0=self.Para0['Initial temperature']
            self.Tempe=np.ones(N)*self.Para0['Background temperature']
            #self.halfwidth=self.Para0['Half width']
            self.yp=np.logspace(start=log(scaleC*c), stop=log(ymax), num=num, base=np.e)-scaleC*c
            self.zp=np.log(c+self.yp)
            self.Tempearr=np.ones([N,num])*self.T0
            self.Tmatrix=np.zeros([num,num])
            self.dTdtmatrix=np.zeros([N,num])
            self.Calc_Tmatrix()


            # self.Ifthermal=self.Para0['If thermal']
            # self.cth=self.Para0['Thermal diffusivity']
            # self.At=self.Para0['Ratio of thermal expansivity to compressibility']
            # self.c=self.Para0['Heat capacity']
            # self.T0=self.Para0['Initial temperature']
            
        traction=[]
        #print('trac_nor',trac_nor)
        for i in range(N):
            if(self.Para0['Vertical principal stress value varies with depth']==True):
                turning_dep=self.Para0['Turnning depth']
                ssv= -self.xg[i,2]/turning_dep+0.2
                if(ssv>1.0):
                    ssv=1.0
                #ssv=ssv*1e6
                ssv=trac_nor*ssv*ssv_scale
            else:
                ssv=trac_nor*ssv_scale
            ssh1=-ssv*ssh1_scale
            ssh2=-ssv*ssh2_scale
            ssv=-ssv
            #ssv= -xg3[i]*maxside/5.;
            #Ph1ang=self.get_rotation1(self.xg[i,0])
            #Ph1ang=np.pi/180.*Ph1ang
            Ph1ang=self.Para0['Angle between ssh1 and X-axis']
            Ph1ang=np.pi/180.*Ph1ang
            v11=cos(Ph1ang)
            v12=-sin(Ph1ang)
            v21=sin(Ph1ang)
            v22=cos(Ph1ang)
            Rmatrix=np.array([[v11,v12],[v21,v22]])
            Pstress=np.array([[ssh1,0],[0,ssh2]])
            stress=np.dot(np.dot(Rmatrix,Pstress),Rmatrix.transpose())
            stress3D=np.array([[stress[0][0],stress[0][1],0],[stress[1][0],stress[1][1],0],[0,0,ssv]])
            
            # asd=np.array([0,1,1])
            # asd1=np.array([0,-1,-1])
            # t1=np.dot(stress3D,asd)
            # t2=np.dot(stress3D,asd1)

            # print(t1-np.dot(t1,asd)*asd,t2-np.dot(t2,asd1)*asd1)
            #project stress tensor into fault surface
            #Me=self.eleVec[i].reshape([3,3])
            #T_global=np.dot(Me.transpose(),T_local)
            tra=np.dot(stress3D,self.eleVec[i,-3:])
            
            #print(tra)
            ev11,ev12,ev13=self.eleVec[i,0],self.eleVec[i,1],self.eleVec[i,2]
            ev21,ev22,ev23=self.eleVec[i,3],self.eleVec[i,4],self.eleVec[i,5]
            ev31,ev32,ev33=self.eleVec[i,6],self.eleVec[i,7],self.eleVec[i,8]
            #print('ev11,ev12,ev13 ',ev11,ev12,ev13)
            #print('ev21,ev22,ev23 ',ev21,ev22,ev23)
            self.Tt1o[i]=tra[0]*ev11+tra[1]*ev12+tra[2]*ev13
            self.Tt2o[i]=tra[0]*ev21+tra[1]*ev22+tra[2]*ev23
            self.Tno[i]=tra[0]*ev31+tra[1]*ev32+tra[2]*ev33

            traction.append(tra-self.Tno[i]*self.eleVec[i,-3:])
            #print(self.Tt1o[i],self.Tt2o[i],self.Tno[i])
            
            solve_normal=self.Para0['Normal traction solved from stress tensor']
            if(solve_normal==False):
                self.Tno[i]=ssv

            
            
        
        self.Tno=np.abs(self.Tno)
        self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
        tem=self.Tt/self.Tno
        x=self.Tt1o/(self.Tt+1e-5)
        y=self.Tt2o/(self.Tt+1e-5)
        solve_shear=self.Para0['Shear traction solved from stress tensor']
        solve_normal=self.Para0['Normal traction solved from stress tensor']
        if(self.Para0['Rake solved from stress tensor']==True):
            self.rake=np.arctan2(y,x)
            print('rake:',self.rake*180.0/np.pi)
        else:
            self.rake=np.ones(len(self.Tt))*self.Para0['Fix_rake']
            self.rake=self.rake/180.0*np.pi
        
        if(solve_normal==False and self.Para0['Vertical principal stress value varies with depth']!=True):
            self.Tno=np.ones(len(self.Tno))*trac_nor
        
        #self.rake=np.ones(len(x))*35.0/180.0*np.pi
        self.vec_Tra=np.array([x,y]).transpose()
        
        #print(self.Tt1o)
        #print(np.max(tem),np.min(tem))
        if(self.Para0['Initlab']==True):
            self.Tn_edge()

        T_globalarr=[]
        N=self.Tt1o.shape[0]
        self.Vpl_con=1e-6
        self.Vpl_con=self.Para0['Plate loading rate']
        self.Grad_slpv_con(const=True)

        self.Vpls=np.zeros(N)
        self.Vpld=np.zeros(N)
        
        self.shear_loadingS=np.zeros(N)
        self.shear_loadingD=np.zeros(N)
        self.shear_loading=np.zeros(N)
        self.normal_loading=np.zeros(N)

        self.V0=self.Para0['Reference slip rate']
        # self.dc=np.ones(N)*0.01
        # self.f0=np.ones(N)*0.4
        self.dc=np.ones(N)*0.02
        self.f0=np.ones(N)*self.Para0['Reference friction coefficient']
        self.a=np.zeros(N)
        self.b=np.ones(N)*0.03

        self.slipv1=np.zeros(N)
        self.slipv2=np.zeros(N)
        #self.slipv=np.ones(N)*self.Vpl_con
        self.slipv=self.slipvC
        self.slip1=np.zeros(N)
        self.slip2=np.zeros(N)
        self.slip=np.zeros(N)

        self.arriT=np.ones(N)*1e9

        
        if(self.InputHetoparamter==True):
            self.read_parameter(self.Para0['Inputparamter file'])
            #print(min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge)
            #self.randompatch()
            self.maxslipv0=np.max(self.slipv)
            self.rake0=np.copy(self.rake)
            self.calc_nucleaszie_cohesivezone()
        else:
            boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(self.elelst)
            boundary_coord=self.nodelst[boundary_nodes-1]
            index_surface=np.where(np.abs(boundary_coord[:,2]-0.0)<1e-5)[0]
            index_b=np.arange(0,len(boundary_coord),1)
            index_sb=np.setdiff1d(index_b,index_surface)
            boundary_coord_surface=boundary_coord[index_surface]
            boundary_coord_sb=boundary_coord[index_sb]
            #print(boundary_coord.shape,boundary_nodes.shape)

            xmin,xmax=np.min(self.xg[:,0]),np.max(self.xg[:,0])
            ymin,ymax=np.min(self.xg[:,1]),np.max(self.xg[:,1])
            zmin,zmax=np.min(self.xg[:,2]),np.max(self.xg[:,2])

            nux=self.Para0['Nuclea_posx']
            nuy=self.Para0['Nuclea_posy']
            nuz=self.Para0['Nuclea_posz']
            nuclearloc=np.array([nux,nuy,nuz])
            #nuclearloc=np.array([-20000,0,-20000])
            Wedge=self.Para0['Widths of VS region']
            Wedge_surface=self.Para0['Widths of surface VS region']
            self.localTra=np.zeros([N,2])
            transregion=self.Para0['Transition region from VS to VW region']
            aVs=self.Para0['Rate-and-state parameters a in VS region']
            bVs=self.Para0['Rate-and-state parameters b in VS region']
            dcVs=self.Para0['Characteristic slip distance in VS region']
            aVw=self.Para0['Rate-and-state parameters a in VW region']
            bVw=self.Para0['Rate-and-state parameters b in VW region']
            dcVw=self.Para0['Characteristic slip distance in VW region']

            aNu=self.Para0['Rate-and-state parameters a in nucleation region']
            bNu=self.Para0['Rate-and-state parameters b in nucleation region']
            dcNu=self.Para0['Characteristic slip distance in nucleation region']
            slivpNu=self.Para0['Initial slip rate in nucleation region']
            Set_nuclear=self.Para0['Set_nucleation']==True
            Radiu_nuclear=self.Para0['Radius of nucleation']
            ChangefriA=self.Para0['ChangefriA']==True

            for i in range(self.Tt1o.shape[0]):
                #tem=min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge
                coords1=np.array([self.xg[i]])
                #print(coords1.shape, boundary_coord.shape)
                distem=find_min_euclidean_distance(coords1, boundary_coord_sb)
                dis=distem/Wedge
                dis_surface=np.copy(dis)
                dis1=(distem-Wedge)/transregion
                if(len(boundary_coord_surface)>10):  #in case there is free surface
                    distem_surface=find_min_euclidean_distance(coords1, boundary_coord_surface)
                    dis_surface=distem_surface/Wedge_surface
                    dis1=min(distem-Wedge,distem_surface-Wedge_surface)/transregion
                nuclearregion=1.0-transregion
                

                if(dis<1.0 or dis_surface<1.0):
                    self.a[i]=aVs
                    self.b[i]=bVs
                    #self.slipv[i]=self.Vpl_con
                    self.dc[i]=dcVs
                
                elif(dis1<1.0):
                    
                    if(ChangefriA==True):
                        self.a[i]=aVs-(aVs-aVw)*dis1
                        self.b[i]=bVs
                    else:
                        self.a[i]=aVs
                        self.b[i]=bVs-(bVs-bVw)*dis1
                    #self.slipv[i]=self.Vpl_con
                    self.dc[i]=dcVs
                    
                
                else:
                    self.a[i]=aVw
                    self.b[i]=bVw
                    self.dc[i]=dcVw
                    #self.slipv[i]=self.Vpl_con
                    # self.Tt1o[i]=self.Tt1o[i]*2
                    # self.Tt2o[i]=self.Tt2o[i]*2
                

                
                
                distem=np.linalg.norm(self.xg[i]-nuclearloc)

                if(distem<Radiu_nuclear and Set_nuclear==True):
                    self.slipv[i]=slivpNu
                    #self.slipv[i]=self.Vpl_con
                    self.dc[i]=dcNu
                    self.a[i]=aNu
                    self.b[i]=bNu


                
                T_local=np.zeros(3)
                T_local[0]=cos(self.rake[i])
                T_local[1]=sin(self.rake[i])
                Me=self.eleVec[i].reshape([3,3])
                T_global=np.dot(Me.transpose(),T_local)
                #print(self.Tt1o[i],self.Tt2o[i],T_global)
                T_globalarr.append(T_global)  
            self.T_globalarr=np.array(T_globalarr)
            
            #print(np.min(self.a))
            # self.Tt1o=self.Tt*np.cos(self.rake)
            # self.Tt2o=self.Tt*np.sin(self.rake)
            self.slipv1=self.slipv*np.cos(self.rake)+1e-16
            self.slipv2=self.slipv*np.sin(self.rake)+1e-16

            if(solve_shear==False):
                self.Tt=(self.Tno-self.P*1e-6)*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.slipvC))/self.a))
                
                #self.Tt=self.Tt*0.1
                #self.Tt1o=self.Tno*self.a*np.arcsinh(self.slipv1/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
                #self.Tt2o=self.Tno*self.a*np.arcsinh(self.slipv2/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
            
            

            x=np.cos(self.rake)
            y=np.sin(self.rake)
            self.vec_Tra=np.array([np.cos(self.rake),np.sin(self.rake)]).transpose()
            
            # x=self.Tt1o/self.Tt
            # y=self.Tt2o/self.Tt
            # self.vec_Tra=np.array([x,y]).transpose()
            #print(self.vec_Tra.shape)

            self.fric=self.Tt/(self.Tno-self.P*1e-6)
            
            self.state=np.log(np.sinh(self.Tt/(self.Tno-self.P*1e-6)/self.a)*2.0*self.V0/self.slipv)*self.a

            #self.Tt=self.Tt*0.98
            self.Tt1o=self.Tt*x
            self.Tt2o=self.Tt*y
            
            #self.state1=np.log(np.sinh(self.Tt1o/self.Tno/self.a)*2.0*self.V0/self.slipv1)*self.a
            #self.state2=np.log(np.sinh(self.Tt2o/self.Tno/self.a)*2.0*self.V0/self.slipv2)*self.a
            #print(np.max(self.state),np.min(self.state))
            self.maxslipv0=np.max(self.slipv)
            self.rake0=np.copy(self.rake)
            self.calc_nucleaszie_cohesivezone()
            #self.randompatch()
        
        #self.getpatchs()
        f=open('Tvalue.txt','w')
        f.write('xg1,xg2,xg3,se1,se2,se3\n')
        for i in range(len(self.xg)):
            f.write('%f,%f,%f,%f,%f,%f\n' %(self.xg[i,0],self.xg[i,1],self.xg[i,2],traction[i][0],traction[i][1],traction[i][2]))
            #f.write('%f,%f,%f,%f,%f,%f\n' %(self.xg[i,0],self.xg[i,1],self.xg[i,2],self.T_globalarr[i,0],self.T_globalarr[i,1],self.T_globalarr[i,2]))
        
        f.close()
        
 
    #read vtk file for initial condition if it start from previous results
    def read_vtk(self,fname):
        #K=450
        #mesh0 = pv.read("examples/case1/out/step%d.vtk"%K)
        mesh0 = pv.read(fname)
        self.rake = mesh0.cell_data['rake[Degree]'].astype(np.float64) / 180.0 * np.pi
        self.Tt1o = mesh0.cell_data['Shear_1[MPa]'].astype(np.float64)
        self.Tt2o = mesh0.cell_data['Shear_2[MPa]'].astype(np.float64)
        self.Tt = mesh0.cell_data['Shear_[MPa]'].astype(np.float64)
        self.Tno = mesh0.cell_data['Normal_[MPa]'].astype(np.float64)

        self.slipv1 = mesh0.cell_data['Slipv1[m/s]'].astype(np.float64)
        self.slipv2 = mesh0.cell_data['Slipv2[m/s]'].astype(np.float64)
        self.slipv = mesh0.cell_data['Slipv[m/s]'].astype(np.float64)

        self.slip = mesh0.cell_data['slip[m]'].astype(np.float64)
        self.slip1 = mesh0.cell_data['slip1[m]'].astype(np.float64)
        self.slip2 = mesh0.cell_data['slip2[m]'].astype(np.float64)

        self.state = mesh0.cell_data['state'].astype(np.float64)
        self.fric = mesh0.cell_data['fric'].astype(np.float64)
        # self.a=mesh0.cell_data['a']
        # self.b=mesh0.cell_data['b']
        # self.dc=mesh0.cell_data['dc']



    #read initial condition from outside files
    def read_parameter(self,fname):
        f=open(self.dirname+'/'+fname,'r')
        values=[]
        for line in f:
            tem=line.split()
            tem=np.array(tem).astype(float)
            values.append(tem)
        f.close()

        values=np.array(values)
        Ncell=self.eleVec.shape[0]
        self.rake=values[:Ncell,0]
        self.a=values[:Ncell,1]
        self.b=values[:Ncell,2]
        self.dc=values[:Ncell,3]
        self.f0=values[:Ncell,4]
        #self.Tt1o=values[:Ncell,5]*1e6
        #self.Tt2o=values[:Ncell,5]*0
        self.Tt=values[:Ncell,5]
        self.Tno=values[:Ncell,6]
        #self.slipv1=values[:Ncell,7]
        #self.slipv2=-values[:Ncell,7]*0.0
        self.slipv=values[:Ncell,7]
        #self.slipv=self.slipvC
        
        self.shear_loading=values[:Ncell,8]
        self.normal_loading=values[:Ncell,9]

        try:
            self.P=values[:Ncell,10]*1e6
            self.hs=values[:Ncell,11]
        except:
            print('No external Initial porepressure and shear zone width data!')
        #     return

        #slipv1=1e-9
        #self.Tt=self.Tno*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
        #self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
        
        # x=self.Tt1o/self.Tt
        # y=self.Tt2o/self.Tt
        # self.rake=np.arctan2(y,x)
        #print(self.rake)
        #self.vec_Tra=np.array([x,y]).transpose()
        #print(self.vec_Tra.shape)
        #self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)

        
        x=np.cos(self.rake)
        y=np.sin(self.rake)
        #self.Tt1o=self.Tt*x
        #self.Tt2o=self.Tt*y+1.0
        self.slipv1=self.slipv*x+1e-16
        self.slipv2=self.slipv*y+1e-16
        
        self.vec_Tra=np.array([x,y]).transpose()
        self.Tt=(self.Tno-self.P*1e-6)*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.slipvC))/self.a))
        #self.Tt=self.Tno*self.a*np.arcsinh(self.slipv/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.slipvC))/self.a))
        #self.Tt1o=self.Tno*self.a*np.arcsinh(self.slipv1/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
        #self.Tt2o=self.Tno*self.a*np.arcsinh(self.slipv2/(2.0*self.V0)*np.exp((self.f0+self.b*np.log(self.V0/self.Vpl_con))/self.a))
        #self.Tt2o=np.zeros(len(self.a))
        self.Tt1o=self.Tt*x
        self.Tt2o=self.Tt*y

        self.fric=self.Tt/(self.Tno-self.P*1e-6)
        self.state=np.log(np.sinh(self.Tt/(self.Tno-self.P*1e-6)/self.a)*2.0*self.V0/self.slipv)*self.a
        #self.state1=np.log(np.sinh(self.Tt1o/self.Tno/self.a)*2.0*self.V0/self.slipv1)*self.a
        #self.state2=np.log(np.sinh((self.Tt2o)/self.Tno/self.a)*2.0*self.V0/(self.slipv2))*self.a
        
        #print(self.state1,self.state2)
        T_globalarr=[]
        for i in range(len(self.rake)):
            T_local=np.zeros(3)
            T_local[0]=cos(self.rake[i])
            T_local[1]=sin(self.rake[i])
            Me=self.eleVec[i].reshape([3,3])
            T_global=np.dot(Me.transpose(),T_local)
            #print(self.Tt1o[i],self.Tt2o[i],T_global)
            T_globalarr.append(T_global)  
        self.T_globalarr=np.array(T_globalarr)

    #Partial derivative calculation
    def derivative_(self,Tno,Tt1o,Tt2o,state):
        Tno=Tno*1e6
        Tt1o=Tt1o*1e6
        Tt2o=Tt2o*1e6
        
        P=self.P[self.local_index]
        dPdt=self.dPdt0[self.local_index]
        AdotV1=self.AdotV1[self.local_index]
        AdotV2=self.AdotV2[self.local_index]
        shear_loading=self.shear_loading[self.local_index]
        dsigmadt=self.dsigmadt[self.local_index]
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

        #theta=dc/V0*np.exp((state-f0)/b)
        #dthetadt=1.0-self.slipv*theta/dc
        #dthetadt=-theta*self.slipv/dc*np.log(theta*self.slipv/dc)
        # slipv1 = self.slipv1
        # slipv2 = self.slipv2
        # slipv_gpu=np.sqrt(slipv1*slipv1+slipv2*slipv2)

        #print('Tno-P:',np.max(Tno-P),np.min(Tno-P))
        
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

    def Calc_Pmatrix(self):
        K=len(self.zp)
        z_k = self.zp
        c_hyd = self.Chyd 
        dz = np.diff(self.zp)
        #for j in range(len(self.eleVec)):
        for i in range(K-1):
            if(i==0):
                self.Pmatrix[0,0]=c_hyd*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
                self.Pmatrix[0,1]=-c_hyd*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
            else:
                self.Pmatrix[i,i]=c_hyd*(np.exp(-(2*z_k[i]-dz[i]/2))+np.exp(-(2*z_k[i]+dz[i]/2)))/(dz[i]*dz[i])
                self.Pmatrix[i,i+1]=-c_hyd*np.exp(-(2*z_k[i]+dz[i]/2))/(dz[i]*dz[i])
                self.Pmatrix[i,i-1]=-c_hyd*np.exp(-(2*z_k[i]-dz[i]/2))/(dz[i]*dz[i])

    def Calc_Tmatrix(self):
        K=len(self.zp)
        z_k = self.zp
        cth = self.cth 
        dz = np.diff(self.zp)
        #for j in range(len(self.eleVec)):
        for i in range(K-1):
            if(i==0):
                self.Tmatrix[0,0]=cth*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
                self.Tmatrix[0,1]=-cth*(np.exp(-(2*z_k[0]-dz[0]/2))+np.exp(-(2*z_k[0]+dz[0]/2)))/(dz[0]*dz[0])
            else:
                self.Tmatrix[i,i]=cth*(np.exp(-(2*z_k[i]-dz[i]/2))+np.exp(-(2*z_k[i]+dz[i]/2)))/(dz[i]*dz[i])
                self.Tmatrix[i,i+1]=-cth*np.exp(-(2*z_k[i]+dz[i]/2))/(dz[i]*dz[i])
                self.Tmatrix[i,i-1]=-cth*np.exp(-(2*z_k[i]-dz[i]/2))/(dz[i]*dz[i])
    
    

    def Calc_P_implicit_mpi(self,dt):
        e = self.DilatancyC  
        
        beta = self.EPermeability    
        c_hyd = self.Chyd
        dc=self.dc[self.local_index]
        f0=self.f0[self.local_index]
        b=self.b[self.local_index]
        slipv=self.slipv[self.local_index]
        h = self.hs[self.local_index]
        P=np.copy(self.P)
        dPdt0=np.copy(self.dPdt0)
        Parr=np.copy(self.Parr)
        z_k = self.zp

        dz = np.diff(self.zp)
        #delta=self.slip
        K=len(self.zp)
        M=self.Pmatrix[:-1,:-1]*dt+np.eye(K-1)
        lu, piv = lu_factor(M)
        #print('self.state_local',np.max(self.state_local))
        theta=dc/self.V0*np.exp((self.state_local-f0)/b)
        dthetadt=1.0-slipv*theta/dc
        #g=e*h*self.slipv/(2.0*beta*c_hyd*dc)*np.log(self.slipv*self.state/dc)*np.exp(-delta/dc)
        #dstatedt = self.b / dc * (self.V0 * np.exp((self.f0 - self.state) / self.b) - self.slipv)
        if(self.Ifcouple==True and self.Ifdila==True):
            g=-e*h/(2.0*beta*c_hyd*theta)*dthetadt
            
            self.porosity[self.local_index]=self.porosity[self.local_index]-e/theta*dthetadt*dt
        elif(self.Ifdila==True and self.Ifthermal==False):
            g=-e*h/(2.0*beta*c_hyd*theta)*dthetadt
            self.porosity[self.local_index]=self.porosity[self.local_index]-e/theta*dthetadt*dt
        else:
            g=dthetadt*0.0
        
        #print('gmax:',np.max(g[self.local_index]),'   gmin:',np.min(g[self.local_index]))
        
        for i in range(len(self.local_index)):
            k=self.local_index[i]
            bv=np.copy(Parr[k,:-1])
            bv[0]=bv[0]-2.0*c_hyd*np.exp(-(z_k[0]-dz[0]/2))*g[i]*dt/dz[0]
            #B1.append(b[0])
            bv[-1]=bv[-1]+dt/(dz[-1]*dz[-1])*c_hyd*np.exp(-(2.0*z_k[-2]+dz[-1]/2))*self.P0
            #couple with temperature
            
            if(self.Ifthermal==True):
               bv=bv+dt*self.At*self.dTdtmatrix[k,:-1]
            x = lu_solve((lu, piv), bv)
            Parr[k,:-1]=np.copy(x)
            #self.dPdt0[i]=(x[0]*1e-6-self.P[i])/dt
            term1 = -np.exp(-(z_k[0] - dz[0]/2)) * (Parr[k,0] - Parr[k,1]+2*dz[0]*g[i]*exp(z_k[0]))
            term2 = np.exp(-(z_k[0] + dz[0]/2)) * (Parr[k,1] - Parr[k,0])
            dPdt0[k]=c_hyd * np.exp(-z_k[0]) * (term1 + term2) / dz[0]**2+self.At*self.dTdtmatrix[k,0]
            #self.dPdt0[i]=0
            P[k]=Parr[k,0]
            
        #print('rank ',rank,np.max(P[self.local_index]),np.min(P[self.local_index]))
        return P,dPdt0,Parr


    def Calc_T_implicit_mpi(self,dt):
        
        trac=np.sqrt(self.Tt1o_local*self.Tt1o_local+self.Tt2o_local*self.Tt2o_local)
        slipv=self.slipv[self.local_index]
        Tempe=np.copy(self.Tempe)
        dTdt0=np.copy(self.dTdt0)
        Tarr=np.copy(self.Tempearr)
        z_k = self.zp

        dz = np.diff(self.zp)
        #delta=self.slip
        K=len(self.zp)
        M=self.Tmatrix[:-1,:-1]*dt+np.eye(K-1)
        lu, piv = lu_factor(M)
        #print('self.state_local',np.max(self.state_local))
        
        #g=e*h*self.slipv/(2.0*beta*c_hyd*dc)*np.log(self.slipv*self.state/dc)*np.exp(-delta/dc)
        #dstatedt = self.b / dc * (self.V0 * np.exp((self.f0 - self.state) / self.b) - self.slipv)
        #g=-trac*1e6*slipv/(2.0*self.c*self.density*self.cth)
        #g=-trac*1e6*slipv/(2.0*self.c*1000*self.cth)
        #g=slipv*0.0
        g=-trac*1e6*slipv/(2.0*self.c*self.cth)
        g=slipv*0.0
        #print('gmax:',np.max(g[self.local_index]),'   gmin:',np.min(g[self.local_index]))
        maxval=0
        maxtra=0
        halfwidth=self.hs*0.5
        for i in range(len(self.local_index)):
            k=self.local_index[i]
            bv=np.copy(Tarr[k,:-1])
            bv[0]=bv[0]-2.0*self.cth*np.exp(-(z_k[0]-dz[0]/2))*g[i]*dt/dz[0]
            #B1.append(b[0])
            bv[-1]=bv[-1]+dt/(dz[-1]*dz[-1])*self.cth*np.exp(-(2.0*z_k[-2]+dz[-1]/2))*self.T0
            
            Ind_term=trac[i]*1e6*slipv[i]/(self.c)*np.exp(-self.yp[:-1]*self.yp[:-1]/(2.0*halfwidth[k]*halfwidth[k]))/(sqrt(2.0*np.pi)*halfwidth[k])
            bv=bv+Ind_term*dt
            if(Ind_term[0]>maxval):
                maxval=Ind_term[0]
                #maxtra=trac[i]*1e6*slipv[i]/(self.c)
            x = lu_solve((lu, piv), bv)
            Tarr[k,:-1]=np.copy(x)
            
            if(self.Ifdila==True):
                self.dTdtmatrix[k,:-1]=np.dot(-self.Tmatrix[:-1,:-1],x)
                self.dTdtmatrix[k,0]=self.dTdtmatrix[k,0]-2.0*self.cth*np.exp(-(z_k[0]-dz[0]/2))*g[i]/dz[0]
                self.dTdtmatrix[k,-2]=self.dTdtmatrix[k,-2]+1.0/(dz[-1]*dz[-1])*self.cth*np.exp(-(2.0*z_k[-2]+dz[-1]/2))*self.T0
                self.dTdtmatrix[k,:-1]=self.dTdtmatrix[k,:-1]+Ind_term
            #self.dPdt0[i]=(x[0]*1e-6-self.P[i])/dt
            term1 = -np.exp(-(z_k[0] - dz[0]/2)) * (Tarr[k,0] - Tarr[k,1]+2*dz[0]*g[i]*exp(z_k[0]))
            term2 = np.exp(-(z_k[0] + dz[0]/2)) * (Tarr[k,1] - Tarr[k,0])
            dTdt0[k]=self.cth * np.exp(-z_k[0]) * (term1 + term2) / dz[0]**2+Ind_term[0]
            #print(np.max(np.exp(-self.yp[:-1]*self.yp[:-1]/(2.0*self.halfwidth*self.halfwidth))))
            
            #self.dPdt0[i]=0
            Tempe[k]=Tarr[k,0]
            #print(Tempe[k])
        #print('maxval:',maxval,maxtra)
        #print('rank ',rank,np.max(P[self.local_index]),np.min(P[self.local_index]))
        return Tempe,dTdt0,Tarr
    

    def init_mpi_local_variables(self):
        if(self.Lt_jud==False):
            self.Tno_local=self.Tno[self.local_index]
            self.Tt1o_local=self.Tt1o[self.local_index]
            self.Tt2o_local=self.Tt2o[self.local_index]
            self.state_local=self.state[self.local_index]
            self.counts = comm.gather(len(self.local_index), root=0)
            self.displs = comm.gather(self.local_index[0], root=0)
            self.index0=np.arange(0,len(self.eleVec),1)
            self.index_ = np.setdiff1d(self.index0, self.local_index)
        else:
            row, col = self.cart_comm.Get_coords(rank)
            if(col==0):
                self.Tno_local=self.Tno[self.local_index]
                self.Tt1o_local=self.Tt1o[self.local_index]
                #print(np.max(self.Tt1o_local),rank)
                self.Tt2o_local=self.Tt2o[self.local_index]
                self.state_local=self.state[self.local_index]
                self.counts = self.diag_comm.gather(len(self.local_index), root=0)
                self.displs = self.diag_comm.gather(self.local_index[0], root=0)
                self.index0=np.arange(0,len(self.eleVec),1)
                self.index_ = np.setdiff1d(self.index0, self.local_index)
            else:
                self.Tno_local=None
                self.Tt1o_local=None
                self.Tt2o_local=None
                self.state_local=None
                self.index0=None
                self.index_ = None
            



    #forward modelling using lattice matrix   
    def simu_forward_mpi_LTM(self,dttry):
        
        cart_rank = self.cart_comm.Get_rank()
        row, col = self.cart_comm.Get_coords(cart_rank)
        # color = 0 if row == col else MPI.UNDEFINED  # 对角线进程 color=0，其他 UNDEFINED
        # diag_comm = comm.Split(color, key=cart_rank)
        #print('start:',self.step,self.Tno.shape,rank,cart_rank)
        slipv1=np.zeros(len(self.slipv1))
        slipv2=np.zeros(len(self.slipv2))
        slipv1[self.local_slipv_index]=self.slipv1[self.local_slipv_index]-self.slipvC[self.local_slipv_index]*np.cos(self.rake0[self.local_slipv_index])
        slipv2[self.local_slipv_index]=self.slipv2[self.local_slipv_index]-self.slipvC[self.local_slipv_index]*np.sin(self.rake0[self.local_slipv_index])
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
        #self.cart_comm.Barrier()
        # if(cart_rank==0 or cart_rank==1 or cart_rank==2 or cart_rank==3):
        #     np.save('AdotV%d'%cart_rank,AdotV1)
        t1 = MPI.Wtime()
        self.compute_time += (t1 - t0)
        
        #row_comm = self.cart_comm.Sub(remain_dims=[False, True])
        #print(row,col,cart_rank,row_comm.rank)
        diag_rank_in_row = 0  # 在行通信器中，对角线进程的局部 rank 就是 row 编号
        M=len(self.xg)
        # 接收缓冲区：所有进程都必须提供，但只有 root 用结果

        sendbuf_v1 = np.array(AdotV1, dtype=np.float64)
        sendbuf_v2 = np.array(AdotV2, dtype=np.float64)
        sendbuf_sig = np.array(dsigmadt, dtype=np.float64)

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
                Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk=self.RungeKutte_solve_Dormand_Prince_(h)
                
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
                        dtnext = max(0.5 * h, dtnext_raw)  # 提前限制
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

                # 5. 所有对角线进程解析广播
                if cart_rank != 0:
                    dtnext = recv_bcast[0]
                    accept = recv_bcast[1]

                # 6. 所有进程统一判断退出
                if accept > 0.5:        # accept == 1.0 → break
                    self.dtnext = dtnext
                    break
                elif accept < -0.5:     # accept == -1.0 → 错误终止
                    if cart_rank == 0:
                        print("Simulation failed: dt too small.")
                    comm.Abort(1)  # stop all pro
                    break
                else:                   # accept == 0.0 → 拒绝，继续循环
                    h = dtnext  # 更新步长
            
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

            if(self.step%self.Para0['outsteps']==0):
                #update slip
                self.slip1=self.slip1+self.slipv1*h
                self.slip2=self.slip2+self.slipv2*h
                self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)
                
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
                    #self.rake=np.arctan2(self.Tt2o,self.Tt1o)
                    self.rake=np.arctan2(self.slip2,self.slip1)
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
    

    #forward modelling
    def simu_forward_mpi_(self,dttry):
        #print('self.P',np.max(self.P))
        slipv1=self.slipv1-self.slipvC*np.cos(self.rake0)
        slipv2=self.slipv2-self.slipvC*np.sin(self.rake0)
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
        t1 = MPI.Wtime()
        self.compute_time += (t1 - t0)

        #Combine results from all ranks
        t0 = MPI.Wtime()
        self.dsigmadt=comm.allreduce(dsigmadt, op=MPI.SUM)
        self.AdotV1=comm.allreduce(AdotV1, op=MPI.SUM)
        self.AdotV2=comm.allreduce(AdotV2, op=MPI.SUM)
        t1 = MPI.Wtime()
        self.comm_time += (t1 - t0)
        #print(np.max(self.AdotV1),rank)

        #comm.Barrier()
        #if(rank==0):
        #    print(self.AdotV2[100:120])

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
        Tno_yhk[Tno_yhk<0.1]=0.1 #constrain normal stress above 0.1 MPa
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
                #self.rake=np.arctan2(self.Tt2o,self.Tt1o)
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
    




    
      
    #RungeKutte iteration
    def RungeKutte_solve_Dormand_Prince_(self,h):
        B21=.2
        B31=3./40
        B32=9./40.

        B41=44./45.
        B42=-56./15
        B43=32./9

        B51=19372./6561.
        B52=-25360/2187.
        B53=64448./6561.
        B54=-212./729.

        B61=9017./3168.
        B62=-355./33.
        B63=-46732./5247.
        B64=49./176.
        B65=-5103./18656.

        B71=35./384.
        B73=500./1113.
        B74=125./192.
        B75=-2187./6784.
        B76=11./84.

        B81=5179./57600.
        B83=7571./16695.
        B84=393./640.
        B85=-92097./339200.
        B86=187./2100.
        B87=1./40.

        Tno=self.Tno_local
        Tt1o=self.Tt1o_local
        Tt2o=self.Tt2o_local
        state=self.state_local
        #P=self.P

        dstatedt1,dsigmadt1,dtau1dt1,dtau2dt1=self.derivative_(Tno,Tt1o,Tt2o,state)
        
        
        #state2=self.state2_gpu
        Tno_yhk=Tno+h*B21*dsigmadt1
        Tt1o_yhk=Tt1o+h*B21*dtau1dt1
        Tt2o_yhk=Tt2o+h*B21*dtau2dt1
        #Tt2o_yhk=Tt2o+h*B21*dtau2dt1
        state_yhk=state+h*B21*dstatedt1
        #P_yhk=P+h*B21*dPdt1
        #print('Tt_yhk',np.mean(Tt_yhk))
        #print(np.max(Tt1o_yhk),np.max(Tt1o),rank)
        dstatedt2,dsigmadt2,dtau1dt2,dtau2dt2=self.derivative_(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        
        Tno_yhk=Tno+h*(B31*dsigmadt1+B32*dsigmadt2)
        Tt1o_yhk=Tt1o+h*(B31*dtau1dt1+B32*dtau1dt2)
        Tt2o_yhk=Tt2o+h*(B31*dtau2dt1+B32*dtau2dt2)
        #Tt2o_yhk=Tt2o+h*(B31*dtau2dt1+B32*dtau2dt2)
        state_yhk=state+h*(B31*dstatedt1+B32*dstatedt2)
        #P_yhk=P+h*(B31*dPdt1+B32*dPdt2)
        #state2_yhk=state2+h*(B31*dstate2dt1+B32*dstate2dt2)
        #print('Tt_yhk',np.mean(Tt_yhk))
        
        dstatedt3,dsigmadt3,dtau1dt3,dtau2dt3=self.derivative_(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B41*dsigmadt1+B42*dsigmadt2+B43*dsigmadt3)
        Tt1o_yhk=Tt1o+h*(B41*dtau1dt1+B42*dtau1dt2+B43*dtau1dt3)
        Tt2o_yhk=Tt2o+h*(B41*dtau2dt1+B42*dtau2dt2+B43*dtau2dt3)
        state_yhk=state+h*(B41*dstatedt1+B42*dstatedt2+B43*dstatedt3)
        #P_yhk=P+h*(B41*dPdt1+B42*dPdt2+B43*dPdt3)
        #state2_yhk=state2+h*(B41*dstate2dt1+B42*dstate2dt2+B43*dstate2dt3)
        #print('Tt_yhk',np.mean(Tt_yhk))

        dstatedt4,dsigmadt4,dtau1dt4,dtau2dt4=self.derivative_(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B51*dsigmadt1+B52*dsigmadt2+B53*dsigmadt3+B54*dsigmadt4)
        Tt1o_yhk=Tt1o+h*(B51*dtau1dt1+B52*dtau1dt2+B53*dtau1dt3+B54*dtau1dt4)
        Tt2o_yhk=Tt2o+h*(B51*dtau2dt1+B52*dtau2dt2+B53*dtau2dt3+B54*dtau2dt4)
        state_yhk=state+h*(B51*dstatedt1+B52*dstatedt2+B53*dstatedt3+B54*dstatedt4)
        #P_yhk=P+h*(B51*dPdt1+B52*dPdt2+B53*dPdt3+B54*dPdt4)
        #state2_yhk=state2+h*(B51*dstate2dt1+B52*dstate2dt2+B53*dstate2dt3+B54*dstate2dt4)

        dstatedt5,dsigmadt5,dtau1dt5,dtau2dt5=self.derivative_(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B61*dsigmadt1+B62*dsigmadt2+B63*dsigmadt3+B64*dsigmadt4+B65*dsigmadt5)
        Tt1o_yhk=Tt1o+h*(B61*dtau1dt1+B62*dtau1dt2+B63*dtau1dt3+B64*dtau1dt4+B65*dtau1dt5)
        Tt2o_yhk=Tt2o+h*(B61*dtau2dt1+B62*dtau2dt2+B63*dtau2dt3+B64*dtau2dt4+B65*dtau2dt5)
        state_yhk=state+h*(B61*dstatedt1+B62*dstatedt2+B63*dstatedt3+B64*dstatedt4+B65*dstatedt5)
        #P_yhk=P+h*(B61*dPdt1+B62*dPdt2+B63*dPdt3+B64*dPdt4+B65*dPdt5)
        #state2_yhk=state2+h*(B61*dstate2dt1+B62*dstate2dt2+B63*dstate2dt3+B64*dstate2dt4+B65*dstate2dt5)
        

        dstatedt6,dsigmadt6,dtau1dt6,dtau2dt6=self.derivative_(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B71*dsigmadt1+B73*dsigmadt3+B74*dsigmadt4+B75*dsigmadt5+B76*dsigmadt6)
        Tt1o_yhk=Tt1o+h*(B71*dtau1dt1+B73*dtau1dt3+B74*dtau1dt4+B75*dtau1dt5+B76*dtau1dt6)
        Tt2o_yhk=Tt2o+h*(B71*dtau2dt1+B73*dtau2dt3+B74*dtau2dt4+B75*dtau2dt5+B76*dtau2dt6)
        state_yhk=state+h*(B71*dstatedt1+B73*dstatedt3+B74*dstatedt4+B75*dstatedt5+B76*dstatedt6)
        #P_yhk=P+h*(B71*dPdt1+B73*dPdt3+B74*dPdt4+B75*dPdt5+B76*dPdt6)
        #print('dstatedt6',np.max(dstatedt6),np.min(dstatedt6))
        #state2_yhk=state2+h*(B71*dstate2dt1+B73*dstate2dt3+B74*dstate2dt4+B75*dstate2dt5+B76*dstate2dt6)
        
        dstatedt7,dsigmadt7,dtau1dt7,dtau2dt7=self.derivative_(Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk)
        Tno_yhk8=Tno+h*(B81*dsigmadt1+B83*dsigmadt3+B84*dsigmadt4+B85*dsigmadt5+B86*dsigmadt6+B87*dsigmadt7)
        Tt1o_yhk8=Tt1o+h*(B81*dtau1dt1+B83*dtau1dt3+B84*dtau1dt4+B85*dtau1dt5+B86*dtau1dt6+B87*dtau1dt7)
        Tt2o_yhk8=Tt2o+h*(B81*dtau2dt1+B83*dtau2dt3+B84*dtau2dt4+B85*dtau2dt5+B86*dtau2dt6+B87*dtau2dt7)
        state_yhk8=state+h*(B81*dstatedt1+B83*dstatedt3+B84*dstatedt4+B85*dstatedt5+B86*dstatedt6+B87*dstatedt7)
        #P_yhk8=P+h*(B81*dPdt1+B83*dPdt3+B84*dPdt4+B85*dPdt5+B86*dPdt6+B87*dPdt7)
        
        #state2_yhk8=state2+h*(B81*dstate2dt1+B83*dstate2dt3+B84*dstate2dt4+B85*dstate2dt5+B86*dstate2dt6+B87*dstate2dt7)

        #state1_yhk_err=cp.abs(state1_yhk8-state1_yhk)
        state_yhk_err=np.abs(state_yhk8-state_yhk)
        #Tno_yhk_err=np.abs(Tno_yhk8-Tno_yhk)
        #P_yhk_err=np.abs(P_yhk8-P_yhk)
        Tt1o_yhk_err=np.abs(Tt1o_yhk8-Tt1o_yhk)
        Tt2o_yhk_err=np.abs(Tt2o_yhk8-Tt2o_yhk)
        


        self.Relerrormax1=np.max(np.abs(state_yhk_err/state_yhk8))+1e-10

        
        Relerrormax2=np.max(np.abs(Tt1o_yhk_err/Tt1o_yhk8))
        Relerrormax2o=np.max(np.abs(Tt2o_yhk_err/Tt2o_yhk8))
        #Relerrormax2_=np.max(np.abs(Tno_yhk_err/Tno_yhk8))
        #Relerrormax2_P=np.max(np.abs(P_yhk_err/P_yhk8))
        #print('error: ',np.max(Relerrormax2),np.max(Relerrormax2o),np.max(Relerrormax2_),np.max(Relerrormax2_P))

        #self.Relerrormax2=max(Relerrormax2,Relerrormax2o,Relerrormax2_,Relerrormax2_P)+1e-10
        self.Relerrormax2=max(Relerrormax2,Relerrormax2o)+1e-10
        #self.Relerrormax2=cp.linalg.norm(Tt1o_yhk_err/Tt1o_yhk8)+1e-10

        #print('errormax1,errormax2,relaemax1,relaemax2:',errormax1,errormax2,self.Relerrormax1,self.Relerrormax2)

        # if((self.maxslipv0)>1e-6):
        #     self.RelTol1=1e-4
        #     self.RelTol2=1e-4
        # else:
        #     self.RelTol1=2e-6
        #     self.RelTol2=2e-6
        
        

        #print(self.Relerrormax1,self.Relerrormax2)

        


        return Tno_yhk,Tt1o_yhk,Tt2o_yhk,state_yhk
    

    #def ouputVTK(self,**kwargs):
    def writeVTK(self,fname):
        Nnode=self.nodelst.shape[0]
        Nele=self.elelst.shape[0]
        f=open(fname,'w')
        f.write('# vtk DataFile Version 3.0\n')
        f.write('test\n')
        f.write('ASCII\n')
        f.write('DATASET  UNSTRUCTURED_GRID\n')
        f.write('POINTS '+str(Nnode)+' float\n')
        for i in range(Nnode):
            f.write('%f %f %f\n'%(self.nodelst[i][0],self.nodelst[i][1],self.nodelst[i][2]))
        f.write('CELLS '+str(Nele)+' '+str(Nele*4)+'\n')
        for i in range(Nele):
            f.write('3 %d %d %d\n'%(self.elelst[i][0]-1,self.elelst[i][1]-1,self.elelst[i][2]-1))
        f.write('CELL_TYPES '+str(Nele)+'\n')
        for i in range(Nele):
            f.write('5 ')
        f.write('\n')
        

        f.write('CELL_DATA %d ' %(Nele))
        f.write('SCALARS Normal_[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tno)):
            f.write('%f '%(self.Tno[i]))
        f.write('\n')
        f.write('SCALARS Pore_pressure[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.P)):
            f.write('%f '%(self.P[i]*1e-6))
        f.write('\n')


        f.write('SCALARS Shear_[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt[i]))
        f.write('\n')

        f.write('SCALARS Shear_1[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt1o[i]))
        f.write('\n')

        f.write('SCALARS Shear_2[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt2o[i]))
        f.write('\n')

        f.write('SCALARS rake[Degree] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.rake)):
            f.write('%f '%(self.rake[i]*180./np.pi))
        f.write('\n')


        f.write('SCALARS state float\nLOOKUP_TABLE default\n')
        for i in range(len(self.state)):
            f.write('%f '%(self.state[i]))
        f.write('\n')


        f.write('SCALARS Slipv[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%.40f '%(self.slipv[i]))
        f.write('\n')

        f.write('SCALARS Slipv1[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%.40f '%(self.slipv1[i]))
        f.write('\n')

        f.write('SCALARS Slipv2[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%.40f '%(self.slipv2[i]))
        f.write('\n')

        f.write('SCALARS a float\nLOOKUP_TABLE default\n')
        for i in range(len(self.a)):
            f.write('%f '%(self.a[i]))
        f.write('\n')

        f.write('SCALARS b float\nLOOKUP_TABLE default\n')
        for i in range(len(self.b)):
            f.write('%f '%(self.b[i]))
        f.write('\n')

        f.write('SCALARS a-b float\nLOOKUP_TABLE default\n')
        for i in range(len(self.b)):
            f.write('%f '%(self.a[i]-self.b[i]))
        f.write('\n')

        f.write('SCALARS dc float\nLOOKUP_TABLE default\n')
        for i in range(len(self.dc)):
            f.write('%.10f '%(self.dc[i]))
        f.write('\n')

        f.write('SCALARS fric float\nLOOKUP_TABLE default\n')
        for i in range(len(self.fric)):
            f.write('%f '%(self.fric[i]))
        f.write('\n')


        f.write('SCALARS slip float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%.20f '%(self.slip[i]))
        f.write('\n')

        f.write('SCALARS slip1 float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%.20f '%(self.slip1[i]))
        f.write('\n')

        f.write('SCALARS slip2 float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%.20f '%(self.slip2[i]))
        f.write('\n')

        f.write('SCALARS slip_plate float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipvC)):
            f.write('%.15f '%(self.slipvC[i]))
        f.write('\n')
        f.close()

    

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

        

    def get_value(self,x,y,z):
        Radius=np.linalg.norm(self.xg-np.array([x,y,z]),axis=1)
        index1_source = np.argsort(Radius)[0]
        return index1_source



    def outputtxt(self,fname):
        directory='out_txt'
        if not os.path.exists(directory):
            os.mkdir(directory)

        xmin,xmax=np.min(self.xg[:,0]),np.max(self.xg[:,0])
        zmin,zmax=np.min(self.xg[:,2]),np.max(self.xg[:,2])
        X1=np.linspace(xmin+self.maxsize,xmax-self.maxsize,500)
        Y1=np.linspace(zmin+self.maxsize,zmax-self.maxsize,300)
        #for i in range(self.xg):
        X_grid, Y_grid = np.meshgrid(X1, Y1)
        X=X_grid.flatten()
        Y=Y_grid.flatten()
        mesh1 = np.column_stack((X, Y))
        #print(self.xg[[0,2]].shape, self.slipv.shape)
        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        #plt.pcolor(slipv_mesh)
        #plt.show()
        f=open(directory+'/X_grid.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%f '%X_grid[i][j])
            f.write('\n')
        f.close()

        f=open(directory+'/Y_grid.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%f '%Y_grid[i][j])
            f.write('\n')
        f.close()


        f=open(directory+'/'+fname+'slipv'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()


        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv1, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'slipv1'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv2, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'slipv2'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        slipv_mesh=griddata(self.xg[:,[0,2]], self.Tt, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'Traction'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        # plt.pcolor(slipv_mesh)
        # plt.show()





            
        
        







        

    

