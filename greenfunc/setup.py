from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

import os
import sys

# Native dependency locations. EIGEN3_INCLUDE_DIR can override detection.
eigen_candidates = [
    os.environ.get("EIGEN3_INCLUDE_DIR"),
    os.path.join(sys.prefix, "include", "eigen3"),
    os.path.join(sys.prefix, "Library", "include", "eigen3"),
    "/usr/include/eigen3",
]
eigen_include_path = next(
    (path for path in eigen_candidates if path and os.path.isdir(path)),
    eigen_candidates[1],
)

if sys.platform == "darwin":
    omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    omp_link_args = ["-lomp"]
elif sys.platform.startswith("linux"):
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]
elif sys.platform == "win32":
    omp_compile_args = ["/openmp"]
    omp_link_args = []
else:
    omp_compile_args = []
    omp_link_args = []

ext_modules = [

    # ------------------------------------------------------------
    # Laplace
    # ------------------------------------------------------------
    Pybind11Extension(
        "regularized_greenfunctions_lib.laplace_lib",
        ["src/laplace_potential.cpp"],
        libraries=["gsl", "gslcblas"],
        cxx_std=17,
    ),

    # ------------------------------------------------------------
    # Kelvin stress
    # ------------------------------------------------------------
    Pybind11Extension(
        "regularized_greenfunctions_lib.kelvin_stress",
        ["src/kelvin_stress.cpp"],
        include_dirs=[eigen_include_path],
        libraries=["gsl", "gslcblas"],
        cxx_std=17,
    ),

    # ------------------------------------------------------------
    # Kelvin stress potential
    # ------------------------------------------------------------
    Pybind11Extension(
        "regularized_greenfunctions_lib.kelvin_stress_potential",
        ["src/stress_potential.cpp"],
        include_dirs=[eigen_include_path],
        libraries=["gsl", "gslcblas"],
        cxx_std=17,
    ),

    # ------------------------------------------------------------
    # Dislocation
    # ------------------------------------------------------------
    Pybind11Extension(
        "regularized_greenfunctions_lib.dislocation_lib",
        ["src/dislocation_stress1.cpp"],
        include_dirs=[eigen_include_path],
        libraries=["gsl", "gslcblas"],
        extra_compile_args=omp_compile_args,
        extra_link_args=omp_link_args,
        cxx_std=17,
    ),
    
    Pybind11Extension(
            "greenfunctions_lib.dislocation_stress",
            ["src/TDstressFS_C.cpp"],
            include_dirs=[eigen_include_path],
            libraries=["gsl", "gslcblas"],
            extra_compile_args=omp_compile_args,
            extra_link_args=omp_link_args,
            cxx_std=17,
        ),

    Pybind11Extension(
        "greenfunctions_lib.dislocation_displacement",
        ["src/TDdispFS_C.cpp"],
        extra_compile_args=omp_compile_args,
        extra_link_args=omp_link_args,
        cxx_std=17,
    ),

    # ------------------------------------------------------------
    # Block ACA + triangular traction Green function
    # ------------------------------------------------------------
    Pybind11Extension(
        "block_aca_lib",
        [
            "src/triangle_aca_py.cpp",
            "src/dislocation_stress1.cpp",
            "src/TDstressFS_C.cpp",
        ],
        include_dirs=[
            eigen_include_path,
        ],
        libraries=[
            "gsl",
            "gslcblas",
        ],
        extra_compile_args=omp_compile_args,
        extra_link_args=omp_link_args,
        cxx_std=17,
    ),

    Pybind11Extension(
        "volume_stress_lib",
        ["src/tetrahedron_stress.cpp"],
        include_dirs=[eigen_include_path],
        libraries=["gsl", "gslcblas"],
        define_macros=[
            ("TETRAHEDRON_STRESS_PYBIND", "1"),
        ],
        extra_compile_args=omp_compile_args,
        extra_link_args=omp_link_args,
        cxx_std=17,
        ),
]


setup(
    name="greenfunctions_lib",
    version="0.1.0",

    packages=find_packages(),

    ext_modules=ext_modules,

    cmdclass={
        "build_ext": build_ext,
    },

    zip_safe=False,
)

