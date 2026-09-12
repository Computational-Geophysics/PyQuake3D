"""Exercise both Block ACA builders and the ACA result matvec function."""

import numpy as np

from _common import (VECTOR_LABELS, example_arguments, plot_contours,
                     receiver_grid, receiver_triangles, source_triangle)


def _options(block_aca_lib):
    options = block_aca_lib.ACAOptions()
    options.epsilon = 1.0e-6
    options.max_block_rank = 1  # There is exactly one source block.
    options.initial_row = 0
    options.pivot_rtol = 1.0e-12
    return options


def _report(name, result):
    print(f"{name}: block_rank={result.block_rank}, "
          f"A.shape={result.A.shape}, B.shape={result.B.shape}")


def main():
    args = example_arguments(__doc__)
    import block_aca_lib

    receivers, _, _ = receiver_grid(args.grid_size)
    observation_mesh = receiver_triangles(receivers)
    triangle = source_triangle()
    sources = triangle[None, :, :]
    nu, mu = 0.25, 30.0e9
    source_slip = np.array([1.0, 0.35, 0.15])

    regularized = block_aca_lib.compute_triangle_aca(
        observation_mesh, sources, nu, mu, 0.05, _options(block_aca_lib))
    analytical = block_aca_lib.compute_triangle_aca_analytical(
        observation_mesh, sources, nu, mu, True, _options(block_aca_lib))

    for name, result, filename in (
        ("regularized ACA", regularized, "block_aca_regularized.png"),
        ("analytical ACA", analytical, "block_aca_analytical.png"),
    ):
        _report(name, result)
        # matvec evaluates the compressed operator for one source-slip vector.
        values = np.asarray(result.matvec(source_slip)).reshape(-1, 3)
        plot_contours(receivers, values, VECTOR_LABELS, name, filename,
                      triangle, args.show)


if __name__ == "__main__":
    main()
