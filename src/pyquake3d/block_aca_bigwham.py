"""Block adaptive cross approximation inspired by BigWham.

The matrix is viewed as an ``n_block_rows x n_block_cols`` array of
``p x p`` blocks.  The routine never needs to assemble the full matrix when a
block callback is supplied.

BigWham reference algorithm:
    M ~= A @ B
    A_k = R[:, j_k]
    B_k = inv(R[i_k, j_k]) @ R[i_k, :]

This implementation uses ``numpy.linalg.solve`` instead of forming the pivot
inverse explicitly, which is numerically preferable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BlockMatrix(Protocol):
    """Minimal matrix-free interface required by block ACA."""

    n_block_rows: int
    n_block_cols: int
    block_size: int

    def block(self, i: int, j: int) -> NDArray: ...


class DenseBlockMatrix:
    """Expose a dense scalar matrix through the block-matrix interface."""

    def __init__(self, matrix: ArrayLike, block_size: int):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if matrix.shape[0] % block_size or matrix.shape[1] % block_size:
            raise ValueError("both matrix dimensions must be divisible by block_size")
        self.matrix = matrix
        self.block_size = int(block_size)
        self.n_block_rows = matrix.shape[0] // block_size
        self.n_block_cols = matrix.shape[1] // block_size

    def block(self, i: int, j: int) -> NDArray:
        p = self.block_size
        return self.matrix[i * p : (i + 1) * p, j * p : (j + 1) * p]


class CallbackBlockMatrix:
    """Matrix-free block matrix backed by ``get_block(i, j)``."""

    def __init__(
        self,
        n_block_rows: int,
        n_block_cols: int,
        block_size: int,
        get_block: Callable[[int, int], ArrayLike],
        dtype=np.float64,
    ):
        self.n_block_rows = int(n_block_rows)
        self.n_block_cols = int(n_block_cols)
        self.block_size = int(block_size)
        self.get_block = get_block
        self.dtype = np.dtype(dtype)

    def block(self, i: int, j: int) -> NDArray:
        value = np.asarray(self.get_block(i, j), dtype=self.dtype)
        expected = (self.block_size, self.block_size)
        if value.shape != expected:
            raise ValueError(f"block({i}, {j}) has shape {value.shape}; expected {expected}")
        return value


@dataclass
class BlockACAResult:
    A: NDArray
    B: NDArray
    block_rank: int
    scalar_rank_bound: int
    pivot_rows: list[int]
    pivot_cols: list[int]
    converged: bool
    stop_reason: str
    estimated_relative_update: float

    def matvec(self, x: ArrayLike) -> NDArray:
        """Apply the compressed matrix without reconstructing it."""
        return self.A @ (self.B @ np.asarray(x))

    def reconstruct(self) -> NDArray:
        """Form the dense approximation (use only for testing/small matrices)."""
        return self.A @ self.B


def _residual_block_row(M: BlockMatrix, A: NDArray, B: NDArray, i: int) -> NDArray:
    """Return R[i, :] = M[i, :] - (A B)[i, :] as a p x (n1*p) array."""
    p, n1 = M.block_size, M.n_block_cols
    row = np.concatenate([np.asarray(M.block(i, j)) for j in range(n1)], axis=1)
    if A.shape[1]:
        row = row - A[i * p : (i + 1) * p, :] @ B
    return row


def _residual_block_column(M: BlockMatrix, A: NDArray, B: NDArray, j: int) -> NDArray:
    """Return R[:, j] = M[:, j] - (A B)[:, j] as an (n0*p) x p array."""
    p, n0 = M.block_size, M.n_block_rows
    column = np.concatenate([np.asarray(M.block(i, j)) for i in range(n0)], axis=0)
    if A.shape[1]:
        column = column - A @ B[:, j * p : (j + 1) * p]
    return column


def _choose_pivot_column(
    row: NDArray, p: int, n1: int, used: set[int], pivot_rtol: float
) -> tuple[Optional[int], float]:
    """Choose the unused block with the largest smallest singular value."""
    candidates: list[tuple[float, int]] = []
    for j in range(n1):
        if j not in used:
            singular_values = np.linalg.svd(row[:, j * p : (j + 1) * p], compute_uv=False)
            candidates.append((float(singular_values[-1]), j))
    if not candidates:
        return None, 0.0
    sigma, j = max(candidates)
    scale = max(float(np.linalg.norm(row, ord="fro")), np.finfo(float).tiny)
    if not np.isfinite(sigma) or sigma <= pivot_rtol * scale:
        return None, sigma
    return j, sigma


def _choose_next_row(column: NDArray, p: int, n0: int, used: set[int]) -> Optional[int]:
    """Choose the unused block row with largest Frobenius norm.

    This is the block analogue of choosing the largest entry in the newly
    sampled residual column.  It plays the role of BigWham's ``searchI0``.
    """
    candidates = [
        (float(np.linalg.norm(column[i * p : (i + 1) * p, :], ord="fro")), i)
        for i in range(n0)
        if i not in used
    ]
    if not candidates:
        return None
    norm, i = max(candidates)
    return i if np.isfinite(norm) and norm > 0.0 else None


def _product_frobenius_sq(A: NDArray, B: NDArray) -> float:
    """Compute ||A B||_F^2 without forming A B."""
    if A.shape[1] == 0:
        return 0.0
    gram_a = A.conj().T @ A
    gram_b = B @ B.conj().T
    return max(0.0, float(np.real(np.trace(gram_a @ gram_b))))


def block_aca(
    matrix: Union[BlockMatrix, ArrayLike],
    block_size: Optional[int] = None,
    *,
    epsilon: float = 1.0e-6,
    max_block_rank: Optional[int] = None,
    initial_row: int = 0,
    pivot_rtol: float = 1.0e-14,
) -> BlockACAResult:
    """Compute a BigWham-style block ACA factorization ``M ~= A @ B``.

    Parameters
    ----------
    matrix:
        A dense 2-D array or an object implementing :class:`BlockMatrix`.
    block_size:
        Block dimension ``p``. Required for a dense input and ignored for a
        block-matrix object.
    epsilon:
        Stop when ``||A_k B_k||_F / ||A B||_F <= epsilon``.  As in BigWham,
        this is an update estimator, not the exact residual norm.
    max_block_rank:
        Maximum number of block updates. The scalar rank is at most
        ``block_size * max_block_rank``.
    initial_row:
        Initial observation block row.
    pivot_rtol:
        Reject numerically singular pivot blocks.
    """
    if isinstance(matrix, np.ndarray) or not hasattr(matrix, "block"):
        if block_size is None:
            raise ValueError("block_size is required for a dense matrix")
        M: BlockMatrix = DenseBlockMatrix(matrix, block_size)
    else:
        M = matrix  # type: ignore[assignment]

    p, n0, n1 = M.block_size, M.n_block_rows, M.n_block_cols
    if not (0 <= initial_row < n0):
        raise ValueError("initial_row is outside the block-row range")
    if epsilon <= 0.0 or pivot_rtol < 0.0:
        raise ValueError("epsilon must be positive and pivot_rtol nonnegative")
    rank_limit = min(n0, n1) if max_block_rank is None else min(max_block_rank, n0, n1)

    sample = np.asarray(M.block(0, 0))
    dtype = np.result_type(sample.dtype, np.float64)
    A = np.empty((n0 * p, 0), dtype=dtype)
    B = np.empty((0, n1 * p), dtype=dtype)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pivot_rows: list[int] = []
    pivot_cols: list[int] = []
    i = initial_row
    relative_update = np.inf
    reason = "maximum block rank reached"

    for _ in range(rank_limit):
        row = _residual_block_row(M, A, B, i).astype(dtype, copy=False)
        j, _ = _choose_pivot_column(row, p, n1, used_cols, pivot_rtol)
        if j is None:
            reason = "no nonsingular unused pivot block"
            break

        pivot = row[:, j * p : (j + 1) * p]
        column = _residual_block_column(M, A, B, j).astype(dtype, copy=False)
        try:
            # Solve pivot @ B_k = row instead of explicitly computing inv(pivot).
            Bk = np.linalg.solve(pivot, row)
        except np.linalg.LinAlgError:
            reason = "singular pivot block"
            break
        if not np.all(np.isfinite(Bk)) or not np.all(np.isfinite(column)):
            reason = "non-finite low-rank factor"
            break

        Ak = column
        A = np.concatenate((A, Ak), axis=1)
        B = np.concatenate((B, Bk), axis=0)
        used_rows.add(i)
        used_cols.add(j)
        pivot_rows.append(i)
        pivot_cols.append(j)

        update_sq = _product_frobenius_sq(Ak, Bk)
        approximation_sq = _product_frobenius_sq(A, B)
        relative_update = np.sqrt(update_sq / approximation_sq) if approximation_sq > 0.0 else 0.0
        if relative_update <= epsilon:
            reason = "relative update tolerance reached"
            break

        next_i = _choose_next_row(column, p, n0, used_rows)
        if next_i is None:
            reason = "no unused residual block row"
            break
        i = next_i

    block_rank = len(pivot_rows)
    converged = reason in {
        "relative update tolerance reached",
        "no nonsingular unused pivot block",
        "no unused residual block row",
    }
    return BlockACAResult(
        A=A,
        B=B,
        block_rank=block_rank,
        scalar_rank_bound=block_rank * p,
        pivot_rows=pivot_rows,
        pivot_cols=pivot_cols,
        converged=converged,
        stop_reason=reason,
        estimated_relative_update=float(relative_update),
    )


def _demo() -> None:
    rng = np.random.default_rng(7)
    p, n0, n1, true_block_rank = 3, 24, 20, 4
    left = rng.standard_normal((n0 * p, true_block_rank * p))
    right = rng.standard_normal((true_block_rank * p, n1 * p))
    dense = left @ right

    result = block_aca(dense, p, epsilon=1.0e-11)
    relative_error = np.linalg.norm(dense - result.reconstruct(), "fro") / np.linalg.norm(dense, "fro")
    x = rng.standard_normal(n1 * p)
    matvec_error = np.linalg.norm(dense @ x - result.matvec(x)) / np.linalg.norm(dense @ x)

    print(f"block rank               : {result.block_rank}")
    print(f"scalar rank upper bound  : {result.scalar_rank_bound}")
    print(f"stop reason              : {result.stop_reason}")
    print(f"exact relative error     : {relative_error:.3e}")
    print(f"relative matvec error    : {matvec_error:.3e}")


    

if __name__ == "__main__":
    _demo()
