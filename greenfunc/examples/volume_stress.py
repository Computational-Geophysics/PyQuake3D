"""Exercise tetrahedral volume stress on an XY receiver surface."""

import numpy as np

from _common import (VOLUME_STRESS_LABELS, example_arguments, plot_contours,
                     receiver_grid)


def main():
    args = example_arguments(__doc__)
    import volume_stress_lib

    receivers, _, _ = receiver_grid(args.grid_size)
    tetrahedron = np.array([
        [-0.60, -0.50, 1.35],
        [0.65, -0.45, 1.25],
        [0.00, 0.70, 1.10],
        [0.00, 0.00, 0.45],
    ])
    eigenstrain = np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    stress = volume_stress_lib.computeStressTetrahedronMixedQuad(
        receivers, tetrahedron, eigenstrain, 30.0e9, 0.25,
        7, 0.01, 3.0)
    # Use the upper triangular face as the projected source marker.
    plot_contours(receivers, stress, VOLUME_STRESS_LABELS,
                  "tetrahedral volume stress", "volume_stress.png",
                  tetrahedron[:3], args.show)


if __name__ == "__main__":
    main()
