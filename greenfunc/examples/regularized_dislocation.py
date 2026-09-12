"""Exercise every function in regularized_greenfunctions_lib.dislocation_lib."""

import numpy as np

from _common import (STRESS_LABELS, VECTOR_LABELS, example_arguments,
                     lame_lambda, plot_contours, receiver_grid, source_triangle)


def main():
    args = example_arguments(__doc__)
    from regularized_greenfunctions_lib import dislocation_lib

    receivers, _, _ = receiver_grid(args.grid_size)
    triangle = source_triangle()
    sources = triangle[None, :, :]
    slip = np.array([[1.0, 0.35, 0.15]])
    eps = np.array([0.05])
    nu, mu = 0.25, 30.0e9
    lam = lame_lambda(mu, nu)

    results = {
        "regularized_dislocation_stress.png": (
            dislocation_lib.compute_dislocation_stresses(
                receivers, sources, slip, nu, mu, 0.05, True)[0],
            STRESS_LABELS, "regularized stress"),
        "regularized_dislocation_stress_eps.png": (
            dislocation_lib.compute_dislocation_stresses_eps(
                receivers, sources, slip, nu, mu, eps, True)[0],
            STRESS_LABELS, "per-source-epsilon stress"),
        "regularized_dislocation_displacement.png": (
            dislocation_lib.compute_dislocation_displacements(
                receivers, sources, slip, lam, mu, 0.05, True)[0],
            VECTOR_LABELS, "regularized displacement"),
        "regularized_dislocation_displacement_eps.png": (
            dislocation_lib.compute_dislocation_displacements_eps(
                receivers, sources, slip, lam, mu, eps, True)[0],
            VECTOR_LABELS, "per-source-epsilon displacement"),
    }
    for filename, (values, labels, title) in results.items():
        plot_contours(receivers, values, labels, title, filename,
                      source=triangle, show=args.show)


if __name__ == "__main__":
    main()
