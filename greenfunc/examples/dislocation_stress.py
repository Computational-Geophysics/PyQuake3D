"""Exercise full-space and half-space analytical dislocation stress."""

from _common import (STRESS_LABELS, example_arguments, lame_lambda, plot_contours,
                     receiver_grid, source_triangle)


def main():
    args = example_arguments(__doc__)
    from greenfunctions_lib import dislocation_stress

    receivers, _, _ = receiver_grid(args.grid_size)
    triangle = source_triangle()
    x, y, z = receivers.T
    p1, p2, p3 = triangle
    ss, ds, ts = 1.0, 0.35, 0.15
    nu, mu = 0.25, 30.0e9
    lam = lame_lambda(mu, nu)

    full_space = dislocation_stress.TDstressFS(
        x, y, z, p1, p2, p3, ss, ds, ts, mu, lam)
    half_space = dislocation_stress.TDstressHS(
        x, y, z, p1, p2, p3, ss, ds, ts, mu, lam)
    plot_contours(receivers, full_space, STRESS_LABELS, "TDstressFS",
                  "dislocation_stress_full_space.png", triangle, args.show)
    plot_contours(receivers, half_space, STRESS_LABELS, "TDstressHS",
                  "dislocation_stress_half_space.png", triangle, args.show)


if __name__ == "__main__":
    main()
