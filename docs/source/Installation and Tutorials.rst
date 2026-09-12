Installation and Tutorials
====================================
We provides a step-by-step guide to use **PyQuake3D**. We'll cover installation, compilation, setup, and running a simple simulation. 

The core code of PyQuake3D no longer supports running on Windows, focusing instead on MPI parallel multi-CPU or multi-GPU operation. 


MPI-Based Execution on Linux
----------------------------------------------------------------------------------------


**Step 1: Set Up Conda environment**

1. Download latest Miniconda3, type:

   ``wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh``

    If you use macOS system, using following instead:

    ``curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh``
2. Install Miniconda with the following command:

   ``bash miniconda.sh -b -u -p ~/miniconda3``

   Parameters:

   - ``-b``: Batch mode, automatically accepts the license agreement.
   - ``-u``: Updates the installation if the directory already exists.
   - ``-p ~/miniconda3``: Specifies the installation directory (here, miniconda3 in the home directory).

3. After installation, initialize Conda for terminal use:

   ``~/miniconda3/bin/conda init``

4. Close and reopen the terminal, or run:

   ``source ~/.bashrc``

5. Verify the Installation, check the Conda version:

   ``conda --version``

6. After installation, create a new environment:

   ``conda create -n PyQuake3D python=3.12``

7. Activate the environment:

   ``conda activate PyQuake3D``

8. Install openmpi via conda:
   ``conda install -c conda-forge openmpi``

9. Install GSL via conda:
   ``conda install -c conda-forge gsl``
.. note::

   Dependencies: Ensure ``wget`` is installed. If not, install it:

   - ``sudo apt install wget``  (For Ubuntu/Debian)
   - ``sudo yum install wget``  (For CentOS/RHEL)


**Step 2: Install Greenfuntions C++ environment**
The operation of pyquake3d relies on Green's functions; the C-language library for Green's functions must be installed.
1. Download from https://github.com/Computational-Geophysics/PyQuake3D

2. Install an MPI implementation and g++ compiler (including gcc and other related tools).
   Common options are:

   - ``sudo apt update``
   - ``sudo apt install g++``
   - ``sudo apt install openmpi-bin libopenmpi-dev``
  
3. Navigate to the ``greenfunc`` directory, run ``python -m pip install -e.`` to build ``greenfuncions_lib``.

.. note::
   - If you us macos system, run ``bash install.sh`` to build ``greenfuncions_lib``.

4. For details on how to call and use the Green's function, you can refer to and run the Python examples provided in the `greenfunc/examples` folder.


   

**Step 3: Install PyQuake3D and C++ Dependencies**


1. Install PyQuake3D using pip:

   ``pip install -e .``

   (make sure PyQuake3D environment is activated).







**Step 4: Examples for MPI-based backend**

.. code-block:: python
   :caption: Examples for MPI-based backend

    import pyquake3d.readmsh as readmsh
    import numpy as np
    import sys
    import matplotlib.pyplot as plt
    import pyquake3d.QDsim as QDsim
    from math import *
    import time
    import argparse
    import os
    import psutil
    from datetime import datetime
    from mpi4py import MPI
    import pyquake3d.config as config
    import pyquake3d.Hmatrix as Hmat

    #Limit threads to 1 to avoid performance issues or resource conflicts caused by multi-threaded parallelism.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    file_name = sys.argv[0]
    print(file_name)

    from config import comm, rank, size

    import sys

    if __name__ == "__main__":
        sim0=None
        fnamePara=None
        HMobj=None

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

                parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
                parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
                parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

                args = parser.parse_args()

                fnamegeo = args.inputgeo
                fnamePara = args.inputpara
            
            except:
                fnamegeo='examples/WMF/WMF20251203.msh'
                fnamePara='examples/WMF/parameter.txt'
            
            
            print('Input msh geometry file:',fnamegeo, flush=True)
            print('Input parameter file:',fnamePara, flush=True)   

            nodelst,elelst=readmsh.read_gmsh(fnamegeo)                             #load mesh file
            Para=config.readPara(fnamePara)                                         #load parameter file
            sim0=QDsim.QDsim(elelst,nodelst,Para)                                   #create earthquake cycle model class
            HMobj=Hmat.Hmatrix(sim0.xg,sim0.nodelst,sim0.elelst,sim0.eleVec,Para)   #create Hmatrice class
            # output intial results
            # sim0.read_vtk('out_vtk/step450.vtu')                                  #load initial file from previous results
            fname='Init.vtu'
            sim0.writeVTU(fname,init=True)                                          #Output initial state
            


        HMobj = comm.bcast(HMobj, root=0)                                           #broadcast Hmatrice class
        sim0 = comm.bcast(sim0, root=0)                                             #broadcast Hmatrice class

        
        sim0.deploy_Hmatrix(HMobj)                                                  #deploy Hmatrice class

        
        
        sim0.calc_greenfuncs_mpi()                                                  #calculate core functions
        
        sim0.start()                                                                #Start forward simulation
        

**MPI parallel running:**

For large-scale simulations using the MPI-parallel H-matrix version, use the following command (not necessary in the root directory):

.. code-block:: bash

    mpirun -n <N> python -m pyquake3d.main_mpi -g <input_geometry_file> -p <input_parameter_file>


For example, using 10 parallel mpi processes on BP5-QD model:

.. code-block:: bash

   mpirun-np 10 python -m pyquake3d.main_mpi -g examples/BP5-QD/bp5t.msh-p examples/BP5-QD/parameter.txt


**MPI-Based multi-GPU Execution on Linux**
------------------------------------------------------------------------------------
First, ensure that you have CUDA correctly installed. Please refer to the GPU acceleration section for detailed instructions.
The startup process for multi-MPI-based multi-GPU versions is basically the same as that for MPI-based CPU versions, 
but with the following differences:the parameters in ``parameter.txt`` must ensure ``GPU:True``, 
and the number of ``GPU_cores`` must not be less than the number of CPU processes. You don't need to specify the 
Max thread workers and Batch_size, as it is only used by the single CPU/GPU version.

For MPI-Based multi-GPU version, use the following command (not necessary in the root directory):

.. code-block:: bash

    mpirun -n <N> python -m pyquake3d.main_gpu_mpi -g <input_geometry_file> -p <input_parameter_file>


For example, using 10 parallel mpi processes on BP5-QD model:

.. code-block:: bash

   mpirun-np 10 python -m pyquake3d.main_gpu_mpi -g examples/BP5-QD/bp5t.msh-p examples/BP5-QD/parameter.txt


.. note::
   - Memory Overhead: To run H-Matrix on a GPU, matrices and vectors must be flattened. 
     The MPI-GPU version requires at least twice the memory of the CPU version due to repeated vector elements.

   - VRAM Allocation: Please verify your GPU's VRAM capacity before starting. 
     PyQuake3D automatically balances the memory load across all detected GPUs to minimize the risk of overflow.

   - Compatibility: PyQuake3D fully supports both multi-GPU and single-GPU acceleration.
  
   - The MPI-based multi-GPU version does not utilize GPU acceleration for fluid-related calculations, but this part of the code is still usable.


Post-Processing
------------------------------------------------------------------------------------------

**Visualization**

Results `vtu` files in dir `out_vtu` can be visualized with `PyVista` and `Paraview`. We provide the `pyquake_tools` code package, which 
is useful for visualizing the results of PyQuake3D, including plotting slip rate, stress, and other variables, as well as creating animations. 
You don't need to install it, just copy the code in `pyquake_tools` to your working directory and import it in your script. 

**Simulation time**

Basic information such as time and step size, as well as the maximum slip rate, is contained in ``state.txt`` to monitor whether the simulation results are normal and to visualize the simulation time.


The total number of rows in ``state.txt`` equals the number of the time steps.Each column is as follows:

.. code-block:: bash

    iteration time_step(s) maximum_strike_slip_rate(m/s) maximum_dip_slip_rate(m/s) time(s) time(d)

