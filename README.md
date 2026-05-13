<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D/raw/main/images/logo/PyQuake3D_grey.png" alt="PyQuake3D logo" style="width:100%;">
</p>

<h1 align="center">PyQuake3D</h1>
<h3 align="center">A Python tool for 3-D earthquake sequence simulations of seismic and aseismic slip</h3>

<p align="center">
  <!-- DOI -->
  <a href="https://doi.org/10.1029/2025JH000871">
    <img src="https://img.shields.io/badge/DOI-10.1029%2F2025JH000871-blue.svg" alt="DOI">
  </a>

  <!-- CuPy support -->
  <img src="https://img.shields.io/badge/GPU%20Acceleration-CuPy-brightgreen.svg" alt="CuPy GPU">

  <!-- MPI support -->
  <img src="https://img.shields.io/badge/Parallel%20Computing-MPI-orange.svg" alt="MPI">

  <!-- Python versions (custom since no PyPI metadata) -->
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg" alt="Python">

  <!-- CUDA (optional but nice for HPC visibility) -->
  <img src="https://img.shields.io/badge/CUDA-11%2B-green.svg" alt="CUDA">
</p>

**PyQuake3D** is a high-performance Python-based Boundary Element Method (BEM) code for simulating sequences of seismic and aseismic slip (SEAS) on a complex 3D fault geometry governed by rate- and state-dependent friction. It combines physics-based modeling with modern parallel computing tools (MPI, GPU acceleration via CuPy) to solve a variety of earthquake cycle and rupture problems. This document provides an overview of how to use the script, as well as a detailed description of the input parameters.

## Authors and Contact

**PyQuake3D** was originally developed by [Dr. Rongjiang Tang](https://scholar.google.com/citations?user=_4cR3zMAAAAJ&hl=zh-CN) and [Dr. Luca Dal Zilio](https://www.lucadalzilio.net/).  
We welcome contributions to the project—please follow the contribution guidelines and help us maintain a clean, consistent codebase.

For questions, suggestions, or collaboration opportunities, feel free to reach out:

-  rongjiang.tang@ntu.edu.sg 
-  luca.dalzilio@ntu.edu.sg

Please refer to the [Code Manual](https://pyquake3d.readthedocs.io/en/latest/) for more details.


<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D/raw/main/images/framework/turkey_displacement.png" alt="Turkey Displacement">
</p>

## Features

-  3D non-planar quasi-dynamic earthquake cycle simulations
-  Support for rate-and-state aging friction laws
-  Support for Hierarchical matrix storage and calculation
-  Support for MPI-based GPU acceleration
-  MPI-based CPU acceleration support
-  Suitable for large model earthquake cycle simulation
-  Support for pore fluid pressure varing with slip due to inelastic processes including dilatancy, pore compaction.
-  Supports fluid thermal pressurization caused by friction heating.

<p align="center">
  <img src="https://github.com/Computational-Geophysics/PyQuake3D//raw/main/images/framework/framework_new.png" alt="Framework Overview">
</p>

## Quick Start
A step by step tutorial on how to install and run [BP5-QD_low_resolution case](tutorials/tutorial_BP5.ipynb) and [circular_asperity_low_resolution case](tutorials/ctutorial_circular_asperity.ipynb). The former uses  parameters to set the initial model, while the latter uses external files to import the initial model.

## Installation

### Python Requirements

PyQuake3D supports Python 3.8 and above, so there is no need to specify any version when installing the dependent libraries.

  Use pip for the quick installation:
```bash
python -m pip install -e .
```
Install cupy if you want to use GPU acceleration, we recommened to use conda (e.g. CUDA 11.8):conda install -c conda-forge cupy cudatoolkit=11.8

### C++ Requirements

The TDstressFS_C.cpp in folder src is a C++ source file that computes Green's functions, translated from the Python script TDstressFS.py to leverage C++'s performance for efficient numerical calculations. It is compiled into a dynamic library, TDstressFS_C.so, using a provided Makefile, which must be executed with the make command before running the code to ensure compatibility across different computing environments. The generated library is called by the Python script Hmatrix.py via dynamic loading (e.g., using ctypes). To use it, navigate to the code directory src, run make to build TDstressFS_C.so.

## Running the Script
PyQuake3D provides two versions of the code, GPU and CPU, which can be run using different main functions：main_gpu_mpi or main_mpi.

## For MPI version, use the following command:
To run the PyQuake3D MPI script, use the following command:
```bash
mpirun -n <N> python -m pyquake3d.main_mpi -g <input_geometry_file> -p <input_parameter_file>
```
Where 10 is the number of virtual cpus. Note that using the mpiexec instead in Windows environment.

For example:
```
To execute benchmarks like BP5-QD, use:
```bash
In the PyQuake3D root directory, To run the BP5-QD benchmark:
mpirun -n 10 python -m pyquake3d.main_mpi -g examples/BP5-QD/bp5t.msh -p examples/BP5-QD/parameter.txt


To run the HF-model:
mpirun -n 10 python -m pyquake3d.main_mpi -g examples/HF-model/HFmodel.msh -p examples/HF-model/parameter.txt

To run the  EAFZ-model:
mpirun -n 10 python -m pyquake3d.main_mpi -g examples/EAFZ-model/turkey.msh -p examples/EAFZ-model/parameter.txt

To run the  Lab-model:
mpirun -n 10 python -m pyquake3d.main_mpi -g examples/Lab-model/lab.msh -p examples/Lab-model/parameter.txt
```

## For MPI GPU version, use the following command:
python -g --inputgeo <input_geometry_file> -p --inputpara <input_parameter_file>
```bash
mpirun -n <N> python -m pyquake3d.main_mpi_gpu -g <input_geometry_file> -p <input_parameter_file>
```
Ensure you modify the input parameter (`parameter.txt`) as follows:
- `GPU`: `True`
- `GPU_cores: `1`

## Parameters Setting
The simulation parameters are implemented by modifying the parameter.txt file, rather than by changing the source code. The heterogeneous stress and friction parameters are imported from external files. Please refer to [Code Manual](https://pyquake3d.readthedocs.io/en/latest/)  for description of parameter details. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
We acknowledge the support and feedback provided by the broader scientific community and all contributors to this work.

Development of the Python-based BEM algorithm was informed by the HBI code introduced in:  
**Ozawa, S., Ida, A., Hoshino, T., & Ando, R.** (2023). *Large-scale earthquake sequence simulations on 3-D non-planar faults using the boundary element method accelerated by lattice H-matrices*. **Geophysical Journal International**, 232(3), 1471–1481.  
[https://doi.org/10.1093/gji/ggad042](https://doi.org/10.1093/gji/ggad042)

The implementation of the stress Green’s functions builds on MATLAB routines from:  
**Nikkhoo, M., & Walter, T. R.** (2015). *Triangular dislocation: an analytical, artefact-free solution*. **Geophysical Journal International**, 201(2), 1119–1141.  
[https://doi.org/10.1093/gji/ggv035](https://doi.org/10.1093/gji/ggv035)

We sincerely thank Ryosuke Ando and So Ozawa for their valuable guidance in the development of the code. We also thank Steffen Börm for his assistance with H-matrix implementation and T. Ben Thompson for his assistance with the H-matrix compression via Adaptive Cross Approximation (ACA).

We gratefully acknowledge the institutional and technical support provided by NTU and EOS, and the contributions of researchers and collaborators who help improve the codebase.

## Examples

Explore selected simulations performed with PyQuake3D:

- [*Seismic cycles on a planar fault with frictional heterogeneity*](https://www.youtube.com/watch?v=N_yA4uY77C0)
- [*Seismic cycles on the East Anatolian Fault Zone*](https://www.youtube.com/watch?v=oFy3FSLs3UQ)

## Citation

If you find this work useful, please cite:

@article{tang2025pyquake3d,<br>
  title={PyQuake3D: A Python tool for 3-D earthquake sequence simulations of seismic and aseismic slip},<br>
  author={Tang, Rongjiang and Gan, Lu and Li, Fusheng and Dal Zilio, Luca},<br>
  journal={Journal of Geophysical Research: Machine Learning and Computation},<br>
  volume={2},<br>
  number={4},<br>
  pages={e2025JH000871},<br>
  year={2025},<br>
  publisher={Wiley Online Library}
}

Tang, R., Gan, L., Li, F., & Dal Zilio, L. (2025). PyQuake3D: A Python tool for 3‐D earthquake sequence simulations of seismic and aseismic slip. Journal of Geophysical Research: Machine Learning and Computation, 2(4), e2025JH000871.



