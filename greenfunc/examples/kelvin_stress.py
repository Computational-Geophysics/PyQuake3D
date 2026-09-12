"""Exercise Kelvin stress and displacement for one loaded triangle."""

import numpy as np

from _common import (STRESS_LABELS, VECTOR_LABELS, example_arguments,
                     lame_lambda, plot_contours, receiver_grid, source_triangle)


def main():
    args = example_arguments(__doc__)
    from regularized_greenfunctions_lib import kelvin_stress

    receivers, _, _ = receiver_grid(args.grid_size)
    triangle = source_triangle()
    sources = triangle[None, :, :]
    force = np.array([[1.0, 0.25, -0.50]])
    nu, mu = 0.25, 30.0e9
    lam = lame_lambda(mu, nu)

    stress = kelvin_stress.compute_kelvin_stresses(
        receivers, sources, force, lam, mu, True)[0]
    displacement = kelvin_stress.compute_kelvin_displacements(
        receivers, sources, force, lam, mu, True)[0]
    plot_contours(receivers, stress, STRESS_LABELS, "Kelvin stress",
                  "kelvin_stress.png", triangle, args.show)
    plot_contours(receivers, displacement, VECTOR_LABELS, "Kelvin displacement",
                  "kelvin_displacement.png", triangle, args.show)


if __name__ == "__main__":
    main()
