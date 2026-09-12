Parameters Setting
========================================
Table parameters setting
---------------------------------------------


The simulation parameters are implemented by modifying the ``parameter.txt`` file, rather than by changing
the source code. The input variable list is in :numref:`General Parameters Settings`. If ``InputHetoparamter`` in :numref:`General Parameters Settings` is ``True``, heterogeneous stress and friction parameters are 
imported from external files. The external filename is defined in ``parameter.txt`` and must remain in the 
same directory as ``parameter.txt``. In this case, you don't need to set the parameters of 
:numref:`Stress and Friction Settings` and :numref:`Nucleation Settings`. Otherwise, you need to appropriately set the parameters of stress and frictional initial conditions shown 
in :numref:`Stress and Friction Settings` and :numref:`Nucleation Settings`. :numref:`Fluid Parameters Settings` and :numref:`Thermal Pressurization Parameter Settings` must be set up if fluid-induced dilatancy and thermal pressurization are considered.


.. _General Parameters Settings:
.. list-table:: General Parameters Settings
   :widths: 28 22 50
   :header-rows: 1
   :align: center

   * - **Parameter**
     - **Default**
     - **Description**
   * - Corefunc directory
     - -
     - The storage path for the kernel function matrix composed of stress Green's functions.
   * - Node_order
     - False
     - If True, the node order of the triangular element is clockwise.
   * - save Corefunc
     - False
     - If True, save the kernel function so that it does not need to be recalculated next time.
   * - Scale_km
     - True
     - If True, coordinates are scaled up by 1000 (modeled in km, suitable for natural earthquakes); otherwise remain in meters (suitable for laboratory earthquakes).
   * - Hmatrix_mpi_plot
     - False
     - Draw H-matrix structure diagram; different colors represent sub-matrices calculated by different processes.
   * - Lattice Matrice
     - False
     - Using Lattice H-Matrice
   * - Lattice Partitioning depth
     - 2
     - The initial Lattice partitioning depth.
   * - Using C++ green function
     - True
     - If True, use C++ to calculate stress Green's functions; otherwise use Python.
   * - GPU
     - False (GPU only)
     - If True, enable GPU parallel acceleration (using ``pytorch``).
   * - GPU_cores
     - 2 (GPU only)
     - Number of GPUs for parallel acceleration (using ``pytorch``).
   * - Max thread workers
     - 50 (single CPU/GPU version only and not updated any more)
     - Number of processors in ProcessPoolExecutor to parallelize Green's function calculations.
   * - Batch_size
     - 1000 (single CPU/GPU version only and not updated any more)
     - Number of batches in ProcessPoolExecutor to parallelize Green's function calculations.
   * - Input Hetoparamter
     - False
     - If True, import heterogeneous stress and friction parameters from external files.
   * - Inputparamter file
     - -
     - File name of imported heterogeneous stress and friction parameters.
   * - Lame constants
     - 0.32 × 10¹¹ Pa
     - First Lamé constant.
   * - Shear modulus
     - 0.32 × 10¹¹ Pa
     - Shear modulus.
   * - Rock density
     - 2670 kg/m³
     - Rock mass density.
   * - Reference slip rate
     - 1 × 10⁻⁶ m/s
     - Reference slip rate.
   * - Reference friction coefficient
     - 0.6
     - Reference friction coefficient.
   * - Plate loading rate
     - 1 × 10⁻⁹ m/s
     - Plate loading rate.

.. note::
   The initial ``Lattice partitioning depth``, typically 2 corresponds to 4*4 grids, and 3 corresponds to 8*8 grids.A larger depth indicates a greater degree of parallelism, but also a greater memory cost. Please ensure that the number of cpus used is an integer multiple of pow(2, depth). For example, if Lattice Partitioning depth=2, then the number of cpus used must be 4, 8, 12, 16, etc.; if Lattice Partitioning depth=3, then the number of cpus used must be 8, 16, 24, 32, etc.


.. _Fluid Parameters Settings:
.. list-table:: Fluid Parameters Settings
   :widths: 28 22 50
   :header-rows: 1
   :align: center

   * - **Parameter**
     - **Default**
     - **Description**
   * - If Dilatancy
     - False
     - If True, implement shear-induced dilatation.
   * - If Coupledthermal
     - True
     - If True, coupled implement shear-induced dilatation and thermal pressurization in earthquake cycle.
   * - Dilatancy coefficient
     - 3 × 10⁻⁴ 
     - Quantifies the volume change (dilation or contraction) in a porous medium due to shear deformation.
   * - Hydraulic diffusivity
     - 1 × 10⁻⁵ m²/s
     - The rate at which pore fluid diffuses across the fault.
   * - Shear zone width
     - 2 × 10⁻³ m
     - The thickness of the actively shearing zone represents the width of the region undergoing significant shear deformation.
   * - Effective compressibility
     - 8 × 10⁻¹¹ Pa⁻¹
     - The volumetric strain response of the porous medium to changes in pore pressure.
   * - Constant background porepressure
     - 2 MPa
     - Constant pore pressure.
   * - Initial porepressure
     - 2 MPa
     - Initial pore pressure.

.. _Thermal Pressurization Parameter Settings:
.. list-table:: Thermal Pressurization Parameter Settings
   :widths: 28 22 50
   :header-rows: 1
   :align: center

   * - **Parameter**
     - **Default**
     - **Description**
   * - If thermal
     - False
     - If True, implement thermal pressurization.
   * - Thermal diffusivity
     - 1 × 10⁻⁶ m²/s
     - Material property that describes how quickly heat spreads through a material.
   * - Shear zone width
     - 2 × 10⁻³ m
     - The thickness of the actively shearing zone represents the width of the region undergoing significant shear deformation.
   * - Ratio of thermal expansivity to compressibility
     - 1 × 10⁻⁵
     - Ratio of thermal expansivity to compressibility.
   * - Heat capacity
     - 3 × 10⁻⁶ J/K
     - A measure of how much heat energy is required to raise the temperature of a system.
   * - Background temperature
     - 0 Celsius degree
     - Background temperature.
   * - Initial temperature
     - 0 Celsius degree
     - Initial temperature.

.. note::

   The implementation of **dilatation** and **thermal pressurization** is controlled
   by the following parameters:

   - **No dilatation or thermal pressurization**
     
     - ``Dilatancy = False``
     - ``Coupledthermal = True`` or ``False``
     - ``thermal = False``

   - **Dilatation only**
     
     - ``Dilatancy = True``
     - ``Coupledthermal = True`` or ``False``
     - ``thermal = False``

   - **Thermal pressurization only**
     
     - ``Dilatancy = True``
     - ``Coupledthermal = False``
     - ``thermal = True``

   - **Coupled dilatation and thermal pressurization**
     
     - ``Dilatancy = True``
     - ``Coupledthermal = True``
     - ``thermal = True``

   - **Limitation**
     
     - The GPU version does not currently include thermal pressurization.


.. _Stress and Friction Settings:
.. list-table:: Stress and Friction Settings
   :widths: 28 18 54
   :header-rows: 1
   :align: center

   * - **Parameter**
     - **Default**
     - **Description**
   * - Half space
     - False
     - If True, calculating half-space Green's functions.
   * - Fix_Tn
     - True
     - If True, fixed the normal stress.
   * - Vertical principal stress (ssv)
     - 1.0
     - The vertical principal stress scale: the real vertical principal stress is obtained by multiplying the scale and the value.
   * - Maximum horizontal principal stress (ssh1)
     - 1.6
     - Maximum horizontal principal stress scale.
   * - Minimum horizontal principal stress (ssh2)
     - 0.6
     - Minimum horizontal principal stress scale.
   * - Angle between ssh1 and X-axis
     - 30°
     - The angle by which the principal stress orientation rotates counterclockwise with the X-axis.
   * - Vertical principal stress value
     - 50 MPa
     - Vertical principal stress value when ``Shear traction solved from stress tensor`` is True, or normal stress when it is False.
   * - Vertical principal stress value varies with depth
     - True
     - If True, vertical principal stress value varies with depth, and the horizontal principal stress value also changes with depth simultaneously.
   * - Turning depth
     - 5000 m
     - If ``Vertical principal stress value varies with depth`` is True, starting at this depth, the stress no longer changes with depth.
   * - Shear traction solved from stress tensor
     - False
     - If True, the non-uniform shear stress is projected onto the non-planar fault surface by the stress tensor.
   * - Rake solved from stress tensor
     - False
     - If True, the non-uniform rakes are solved from the stress tensor.
   * - Fix_rake
     - 30°
     - If True, set fixed rakes when ``Rake solved from stress tensor`` is False.
   * - Widths of VS region
     - 5000 m
     - The width of the velocity-strengthening (VS) region near boundary.
   * - Widths of surface VS region
     - 2000 m
     - Widths of velocity-strengthening (VS) region near free surface.
   * - Transition region from VS to VW region
     - 3000 m
     - Transition region width from velocity-strengthening (VS) to velocity-weakening (VW) region.



.. _Nucleation Settings:
.. list-table:: Nucleation Settings
   :widths: 28 18 54
   :header-rows: 1
   :align: center

   * - **Parameter**
     - **Default**
     - **Description**
   * - Set_nucleation
     - False
     - If True, sets a patch whose shear stress and sliding rate are significantly greater than the surrounding area to meet the nucleation requirements.
   * - Radius of nucleation
     - 8000 m
     - The radius of the nucleation region.
   * - Nuclea_posx
     - 34000 m
     - X-coordinate of the nucleation center.
   * - Nuclea_posy
     - 15000 m
     - Y-coordinate of the nucleation center.
   * - Nuclea_posz
     - -15000 m
     - Z-coordinate of the nucleation center.
   * - Rate-and-state parameters a in VS region
     - 0.04
     - Rate-and-state friction parameter *a* in velocity-strengthening (VS) region.
   * - Rate-and-state parameters b in VS region
     - 0.03
     - Rate-and-state friction parameter *b* in velocity-strengthening (VS) region.
   * - Characteristic slip distance in VS region
     - 0.13 m
     - Characteristic slip distance (*L*) in velocity-strengthening (VS) region.
   * - Rate-and-state parameters a in VW region
     - 0.004
     - Rate-and-state friction parameter *a* in velocity-weakening (VW) region.
   * - Rate-and-state parameters b in VW region
     - 0.03
     - Rate-and-state friction parameter *b* in velocity-weakening (VW) region.
   * - Characteristic slip distance in VW region
     - 0.13 m
     - Characteristic slip distance (*L*) in velocity-weakening (VW) region.
   * - Rate-and-state parameters a in nucleation region
     - 0.004
     - Rate-and-state friction parameter *a* in nucleation region.
   * - Rate-and-state parameters b in nucleation region
     - 0.03
     - Rate-and-state friction parameter *b* in nucleation region.
   * - Characteristic slip distance in nucleation region
     - 0.14 m
     - Characteristic slip distance (*L*) in nucleation region.
   * - Initial slip rate in nucleation region
     - 3e-2 m/s
     - Initial slip rate imposed in the nucleation region.
   * - ChangefriA
     - False
     - If True, the parameter *a* changes gradually while *b* remains unchanged (and vice versa if False).
   * - Initlab
     - True
     - If True, apply random non-uniform normal stress distribution.


.. _Output Settings:
.. list-table:: Output Settings
   :widths: 28 18 54
   :header-rows: 1
   :align: center

   * - **Parameter**
     - **Default**
     - **Description**
   * - totaloutputsteps
     - 2000
     - The total number of calculation time steps.
   * - outsteps
     - 50
     - The time step interval for outputting VTU files.
   * - outputSLIPV
     - False
     - If True, output slip rate data for each recorded step.
   * - outputstu
     - True
     - If True, VTU files will be saved in the outvtk directory.


Fault station output
--------------------
You may ouput the slip rate, shear stress, normal stress, and slip displacement at specific points on the fault surface. 
The output points are defined in the ``faultstation.txt`` file in the parameter directory, which should be placed in the same directory as ``parameter.txt``. The format of each row in the ``faultstation.txt`` file is:
.. code-block:: text

   staion_name(0) X_coordinate(0) Y_coordinate(0) Z_coordinate(0)
   staion_name(1) X_coordinate(1) Y_coordinate(1) Z_coordinate(1)
   …


External file format
--------------------

In addition to ``parameter.txt``, the program requires an external ``.msh`` file for the geometry, as well as input parameter files if ``Input Hetoparamter`` is set to **True**. The filename(s) for these input parameter files are defined within ``parameter.txt``.

The ``.msh`` file contains the necessary mesh data (node coordinates, element connectivity, etc.) and must be exported from **Gmsh** software. Please ensure that you select the **Version 2 / ASCII** format, as this is the only format compatible with the current reader in the code.

When ``Input Hetoparamter`` == **True**, heterogeneous initial conditions and friction/stress parameters can be imported from external file(s). The format of each row in the input parameter file is:

.. code-block:: text

   rake(0) a(0) b(0) dc(0) f0(0) tau(0) sigma(0) vel(0) taudot(0) sigdot(0) InitP(0) hs(0)
   rake(1) a(1) b(1) dc(1) f0(1) tau(1) sigma(1) vel(1) taudot(1) sigdot(1) InitP(1) hs(0)
   …

The total number of rows equals the number of triangular elements (cells) in the mesh.

- **Column 1**: rake angle (in degrees). This value can vary during the simulation.
- **Columns 2-5**: rate-and-state friction law parameters  
  - Column 2: ``a``  
  - Column 3: ``b``  
  - Column 4: characteristic slip distance ``dc`` (or ``L``)  
  - Column 5: reference friction coefficient ``f0``
- **Columns 6-7**: initial traction  
  - Column 6: initial shear traction ``tau``  
  - Column 7: initial normal traction ``sigma`` (positive for compression)
- **Column 8**: initial slip rate ``vel``
- **Columns 9-10**: additional loading rates (beyond the tectonic plate loading)  
  - Column 9: shear traction loading rate ``taudot``  
  - Column 10: normal traction loading rate ``sigdot`` (usually 0)
- **Column 11-12** (optional): initial pore pressure ``InitP`` and shearing zone width ``hs``. If fluid flow is not enabled or these two columns are omitted, a uniform initial setting defined elsewhere will be used.
