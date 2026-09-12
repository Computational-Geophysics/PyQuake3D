import pyquake3d.readmsh as readmsh
import numpy as np
import sys
import matplotlib.pyplot as plt
import pyquake3d.QDsim as QDsim
from math import *
import time
import argparse
import os
from datetime import datetime
from mpi4py import MPI
import pyquake3d.config as config
import pyquake3d.Hmatrix as Hmat


#use only a single thread, thereby avoiding performance degradation.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

file_name = sys.argv[0]
print(file_name)

from pyquake3d.config import comm, rank, size



import sys
#old_stdout = sys.stdout
#log_file = open("message.log", "w")
#sys.stdout = log_file


    

if __name__ == "__main__":

    sim0=None
    fnamePara=None
    HMobj=None
    # jud_coredir=None
    blocks_to_process=[] #Save block submatirx from Hmatrix
    if(rank==0):
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
            fnamegeo='para/BP5-QD/bp5t.msh'
            fnamePara='para/BP5-QD/parameter.txt'
            # fnamegeo='examples/cascadia/50km_43dense_35w.msh'
            # fnamePara='examples/cascadia/parameter.txt'
            # fnamegeo='examples/WMF/WMF20260130.msh'
            # fnamePara='examples/WMF/parameter.txt'
            # fnamegeo='para/tri/triple_junction2.msh'
            # fnamePara='para/tri/parameter.txt'
        
        
        print('Input msh geometry file:',fnamegeo, flush=True)
        print('Input parameter file:',fnamePara, flush=True)   

        #nodelst,elelst=readmsh.read_mshV2(fnamegeo)
        nodelst, elelst = readmsh.read_gmsh(fnamegeo)
        print('elelst',elelst.shape,nodelst.shape)
        Para=config.readPara(fnamePara)
        sim0=QDsim.QDsim(elelst,nodelst,Para)
        HMobj=Hmat.Hmatrix(sim0.xg,sim0.nodelst,sim0.elelst,sim0.eleVec,Para)
        # output intial results
        #sim0.read_vtk('out_vtk/step450.vtu')
        fname=sim0.mpioutdirectory+'/Init.vtu'
        sim0.writeVTU(fname,init=True)
        



    
        

    t0 = MPI.Wtime()
    HMobj = comm.bcast(HMobj, root=0)
    sim0 = comm.bcast(sim0, root=0)
    #bcast_time = (t1 - t0)
    
    sim0.deploy_Hmatrix(HMobj)
    sim0.calc_greenfuncs_mpi()
    
 

    
    sim0.start()
    





