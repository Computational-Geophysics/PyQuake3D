"""Exercise both kernel modes of the Kelvin stress-potential interface."""

from _common import (STRESS_LABELS, example_arguments, lame_lambda, plot_contours,
                     receiver_grid, source_triangle)


def main():
    args = example_arguments(__doc__)
    from regularized_greenfunctions_lib import kelvin_stress_potential

    receivers, _, _ = receiver_grid(args.grid_size)
    triangle = source_triangle()
    sources = triangle[None, :, :]
    nu, mu = 0.25, 30.0e9
    lam = lame_lambda(mu, nu)

    for mode in (0, 1):
        stress = kelvin_stress_potential.compute_stresses_potential(
            receivers, sources, lam, mu, mode)[0]
        plot_contours(receivers, stress, STRESS_LABELS,
                      f"Kelvin stress potential (mode {mode})",
                      f"kelvin_stress_potential_mode_{mode}.png",
                      triangle, args.show)


if __name__ == "__main__":
    main()
