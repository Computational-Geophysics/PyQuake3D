import pyquake3d.readmsh as readmsh
import numpy as np
import sys
import matplotlib.pyplot as plt
import pyquake3d.QDsim_gpu_mpi as QDsim_gpu_mpi
from math import *
import time
import argparse
import os
import psutil
from datetime import datetime
from mpi4py import MPI
import pyquake3d.config as config
import pyquake3d.Hmatrix as Hmat
from pyquake3d.Hmatvec_gpu import BatchedMatVecPreprocessor
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

file_name = sys.argv[0]
print(file_name)

from pyquake3d.config import comm, rank, size

import sys
#old_stdout = sys.stdout
#log_file = open("message.log", "w")
#sys.stdout = log_file

if __name__ == "__main__":
    #eleVec,xg=None,None
    #nodelst,xg=None,None
    sim0=None
    fnamePara=None
    HMobj=None
    # jud_coredir=None
    blocks_to_process=[] #Save block submatirx from Hmatrix
    if(rank==0):
        print('# ----------------------------------------------------------------------------')
        print('# PyQuake3D: Boundary Element Method to simulate sequences of earthquakes and aseismic slips')
        print('# * 3D non-planar quasi-dynamic earthquake cycle simulations')
        print('# * Support for Hierarchical matrix compressed storage and calculation')
        print(f'# * Parallelized with MPI ({size} cpus)')
        print('# * Support for rate-and-state aging friction laws')
        print('# * Supports output to VTU formats')
        print('# * ----------------------------------------------------------------------------')
        try:
            #start_time = time.time()
            parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
            parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
            parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

            args = parser.parse_args()

            fnamegeo = args.inputgeo
            fnamePara = args.inputpara
        
        except:
            # fnamegeo='examples/EAFZ-model/turkey.msh'
            # fnamePara='examples/EAFZ-model/parameter.txt'
            fnamegeo='para/reg/reg1.msh'
            fnamePara='para/reg/parameter.txt'
            # fnamegeo='examples/cascadia/50km_43dense_35w.msh'
            # fnamePara='examples/cascadia/parameter.txt'
            # fnamegeo='examples/WMF/WMF20260201.msh'
            # fnamePara='examples/WMF/parameter.txt'
            # fnamegeo='examples/case1/model1.msh'
            # fnamePara='examples/case1/parameter.txt'
        
        
        print('Input msh geometry file:',fnamegeo, flush=True)
        print('Input parameter file:',fnamePara, flush=True)   

        #nodelst,elelst=readmsh.read_mshV2(fnamegeo)
        nodelst, elelst = readmsh.read_gmsh(fnamegeo)
        Para=config.readPara(fnamePara)
        sim0=QDsim_gpu_mpi.QDsim_gpumpi(elelst,nodelst,Para)
        HMobj=Hmat.Hmatrix(sim0.xg,sim0.nodelst,sim0.elelst,sim0.eleVec,Para)

        
        # output intial results
        #sim0.read_vtk('out_vtk/step450.vtu')
    #     fname='Init.vtu'
    #     sim0.writeVTU(fname,init=True)
        


    HMobj = comm.bcast(HMobj, root=0)
    sim0 = comm.bcast(sim0, root=0)

    
    sim0.deploy_Hmatrix(HMobj)

    
    
    sim0.calc_greenfuncs_mpi()

    # Ne = len(sim0.xg)
    # device = "cuda:0"
    # #device = "cpu"
    # operator = BatchedMatVecPreprocessor(device)

    # # # blocks must be H-matrix blocks created by pyquake3d.Hmatrix
    # operator.transfer_hmatrix(sim0.local_blocks, Ne)
    # X = np.ones((Ne, 2))

    # Y = operator.hmatrix_matvec_2d(X).reshape(Ne, 3)
    # print(torch.max(Y[:,0]),torch.max(Y[:,1]))

    # Y1,Y2,Y3=sim0.tree_block.blocks_process_MVM(X[:,0],X[:,1],sim0.local_blocks)
    # print(np.max(Y1),np.max(Y2))
    
    
    sim0.init_torchtensor()
    
    
    sim0.start_gpu()

    





