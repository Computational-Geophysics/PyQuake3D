"""Exercise both modes of the regularized Laplace interface."""

import numpy as np

from _common import example_arguments, plot_contours, receiver_grid, source_triangle


def main():
    args = example_arguments(__doc__)
    from regularized_greenfunctions_lib import laplace_lib

    receivers, _, _ = receiver_grid(args.grid_size)
    triangle = source_triangle()
    sources = triangle[None, :, :]
    fields = []
    for mode in (0, 1):
        field = laplace_lib.compute_pressure_potential(
            receivers, sources, mode)[0]
        fields.append(field)
    plot_contours(receivers, np.column_stack(fields), ("mode 0", "mode 1"),
                  "Laplace field", "laplace_modes.png", triangle, args.show)


if __name__ == "__main__":
    main()
