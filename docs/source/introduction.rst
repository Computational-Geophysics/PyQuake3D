Introduction
============

**PyQuake3D** is an open-source, Python-based Boundary Element Method (BEM) framework 
designed to simulate sequences of earthquakes and aseismic slip (SEAS) on geometrically 
complex three-dimensional (3D) fault systems governed by rate- and state-dependent 
friction laws. It supports fully arbitrary fault geometries embedded in either a 
uniform elastic half-space or full-space medium, including non-planar surfaces, 
fault stepovers, branches, and roughness. 

This documentation provides an overview of **PyQuake3D**, including its main capabilities, 
usage instructions, and a detailed description of the required input parameters.

The code supports multiple execution backends:

- **GPU-accelerated version**  
  Utilizes Python's ``ProcessPoolExecutor`` for parallel evaluation of Green’s functions 
  (kernels), and leverages GPU-based linear algebra libraries (``CuPy``) to accelerate 
  dense matrix–vector operations.

- **H-matrix + MPI version**  
  Reduces memory footprint and computational cost using hierarchical matrix 
  (H-matrix) compression via Adaptive Cross Approximation (ACA). Distributed 
  parallelism is implemented with MPI (``mpi4py``) to enable efficient large-scale 
  simulations on high-performance computing (HPC) platforms.


**PyQuake3D** is designed to scale from local workstations to HPC clusters, making 
it a flexible and extensible platform for researchers, students, and educators 
interested in exploring the physics of earthquake cycles across a wide range 
of spatial and temporal scales.

PyQuake3D is originally developed and maintained by Rongjiang Tang and Luca Dal Zilio. 
The source code, documentation, and tutorials are hosted on GitHub:

    https://github.com/Computational-Geophysics/PyQuake3D
  

We welcome feedback, testing, and contributions from the broader community.  
For inquiries, collaborations, or support, please contact:

   Luca Dal Zilio (luca.dalzilio@ntu.edu.sg) 

   Rongjiang Tang(rongjiang.igp@hotmail.com)
   
   Earth Observatory of Singapore, Nanyang Technological University
