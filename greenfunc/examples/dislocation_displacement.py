"""Exercise full-space and half-space triangular-dislocation displacement."""

import numpy as np

from _common import (VECTOR_LABELS, example_arguments, plot_contours,
                     receiver_grid, source_triangle)


def main():
    args = example_arguments(__doc__)
    from greenfunctions_lib import dislocation_displacement

    receivers, x_grid, y_grid = receiver_grid(args.grid_size)
    triangle = source_triangle()
    z_grid = np.zeros_like(x_grid)
    p1, p2, p3 = triangle
    ss, ds, ts, nu = 1.0, 0.35, 0.15, 0.25

    functions = (
        ("TDdispFS", dislocation_displacement.TDdispFS,
         "dislocation_displacement_full_space.png"),
        ("TDdispHS", dislocation_displacement.TDdispHS,
         "dislocation_displacement_half_space.png"),
    )
    for name, function, filename in functions:
        components = function(x_grid, y_grid, z_grid, p1, p2, p3,
                              ss, ds, ts, nu)
        values = np.column_stack([np.asarray(value).ravel()
                                  for value in components])
        plot_contours(receivers, values, VECTOR_LABELS, name, filename,
                      triangle, args.show)


if __name__ == "__main__":
    main()
