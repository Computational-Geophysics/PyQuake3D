
try:
    from pyquake3d.config import comm, rank, size
    if(rank==0):
        print('# ----------------------------------------------------------------------------')
        print('# PyQuake3D: Boundary Element Method to simulate sequences of earthquakes and aseismic slips')
        print('# * 3D non-planar quasi-dynamic earthquake cycle simulations')
        print('# * Support for Hierarchical matrix compressed storage and calculation')
        print(f'# * Parallelized with MPI')
        print('# * Support for rate-and-state aging friction laws')
        print('# * Supports output to VTU formats')
        print('# * ----------------------------------------------------------------------------')
except:
    print('No mpi4py')
    print('# ----------------------------------------------------------------------------')
    print('# PyQuake3D: Boundary Element Method to simulate sequences of earthquakes and aseismic slips')
    print('# * 3D non-planar quasi-dynamic earthquake cycle simulations')
    print('# * Support for Hierarchical matrix compressed storage and calculation')
    print(f'# * Parallelized with MPI')
    print('# * Support for rate-and-state aging friction laws')
    print('# * Supports output to VTU formats')
    print('# * ----------------------------------------------------------------------------')



