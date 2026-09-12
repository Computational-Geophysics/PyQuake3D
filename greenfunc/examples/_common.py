"""Shared geometry and plotting utilities for the examples."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
STRESS_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")
VOLUME_STRESS_LABELS = ("11", "12", "13", "22", "23", "33")
VECTOR_LABELS = ("x", "y", "z")


def example_arguments(description):
    parser = ArgumentParser(description=description)
    parser.add_argument("--grid-size", type=int, default=21,
                        help="receiver points along each XY direction (default: 21)")
    parser.add_argument("--show", action="store_true",
                        help="also display figures interactively")
    args = parser.parse_args()
    if args.grid_size < 3:
        parser.error("--grid-size must be at least 3")
    return args


def receiver_grid(grid_size=21, extent=3.0, z=0.0):
    """Return flattened points and the corresponding square mesh."""
    axis = np.linspace(-extent, extent, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(),
                              np.full(x_grid.size, z)))
    return points, x_grid, y_grid


def source_triangle():
    """One buried, gently tilted triangular source."""
    return np.array([
        [-0.65, -0.50, -1.20],
        [0.70, -0.45, -1.05],
        [0.05, 0.75, -0.80],
    ], dtype=float)


def receiver_triangles(points, side=0.08):
    """Create small, upward-oriented triangles centered on XY points."""
    height = np.sqrt(3.0) * side / 2.0
    offsets = np.array([
        [-side / 2.0, -height / 3.0, 0.0],
        [side / 2.0, -height / 3.0, 0.0],
        [0.0, 2.0 * height / 3.0, 0.0],
    ])
    return points[:, None, :] + offsets[None, :, :]


def lame_lambda(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def plot_contours(points, values, labels, title, filename, source=None, show=False):
    """Save component fields as filled contours on the XY receiver plane."""
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != points.shape[0]:
        raise ValueError("values must have one row per receiver point")
    if values.shape[1] != len(labels):
        raise ValueError("the number of labels must match the value components")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(f"{title} returned non-finite values")

    x_coordinates = np.unique(points[:, 0])
    y_coordinates = np.unique(points[:, 1])
    expected_size = x_coordinates.size * y_coordinates.size
    if expected_size != points.shape[0]:
        raise ValueError("contour plots require a complete rectangular XY grid")

    count = values.shape[1]
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4.5 * columns, 3.9 * rows),
                             constrained_layout=True)

    for component, (axis, label) in enumerate(zip(axes.flat, labels)):
        field = values[:, component]
        field_grid = field.reshape(y_coordinates.size, x_coordinates.size)
        contours = axis.contourf(x_coordinates, y_coordinates, field_grid,
                                 levels=40, cmap="coolwarm")
        if source is not None:
            closed = np.vstack((source, source[0]))
            axis.plot(closed[:, 0], closed[:, 1], color="black",
                      linewidth=1.4, label="source projection")
            axis.legend(loc="upper right", fontsize=8)
        axis.set_title(f"{title}: {label}")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_aspect("equal")
        fig.colorbar(contours, ax=axis, shrink=0.82)

    for axis in axes.flat[count:]:
        axis.remove()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / filename
    fig.savefig(output, dpi=180)
    print(f"saved {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output
