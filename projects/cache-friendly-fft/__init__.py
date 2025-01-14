import numpy as np

from transpose import transpose_square
from util import lb_exact


def _interleave(x, scratch):
    """Interleave the elements in an array in-place.

    For example, if `x` is `array([1, 2, 3, 4, 5, 6, 7, 8])`, then its
    contents will be rearranged to `array([1, 5, 2, 6, 3, 7, 4, 8])`.

    `scratch` is an externally-allocated buffer, whose `dtype` matches
    `x` and whose length is at least half the length of `x`.
    """
    assert len(x.shape) == len(scratch.shape) == 1
    
    n, = x.shape
    assert n % 2 == 0

    half_n = n // 2
    assert scratch.shape[0] >= half_n

    assert x.dtype == scratch.dtype
    scratch = scratch[:half_n]
    
    scratch[:] = x[:half_n]  # Save the first half of `x`.
    for i in range(half_n):
        x[2 * i] = scratch[i]
        x[2 * i + 1] = x[half_n + i]


def _deinterleave(x, scratch):
    """Deinterleave the elements in an array in-place.

    For example, if `x` is `array([1, 2, 3, 4, 5, 6, 7, 8])`, then its
    contents will be rearranged to `array([1, 3, 5, 7, 2, 4, 6, 8])`.

    `scratch` is an externally-allocated buffer, whose `dtype` matches
    `x` and whose length is at least half the length of `x`.
    """
    assert len(x.shape) == len(scratch.shape) == 1
    
    n, = x.shape
    assert n % 2 == 0

    half_n = n // 2
    assert scratch.shape[0] >= half_n

    assert x.dtype == scratch.dtype
    scratch = scratch[:half_n]

    for i in range(half_n):
        x[i] = x[2 * i]
        scratch[i] = x[2 * i + 1]
    x[half_n:] = scratch


def _fft_inplace_evenpow(x, scratch):
    """In-place FFT of length 2^even"""
    # Reshape `x` to a square matrix in row-major order.
    vec_len = x.shape[0]
    n = 1 << (lb_exact(vec_len) >> 1)  # Matrix dimension
    x.shape = n, n, 1

    # We want to recursively apply FFT to every column. Because `x` is
    # in row-major order, we transpose it to make the columns contiguous
    # in memory, then recurse, and finally transpose it back. While the
    # row is in cache, we also multiply by the twiddle factors.
    transpose_square(x)
    for i, row in enumerate(x[..., 0]):
        _fft_inplace(row, scratch)
        # Multiply by the twiddle factors
        for j in range(n):
            row[j] *= np.exp(-2j * np.pi * (i * j) / vec_len)
    transpose_square(x)

    # Now recursively apply FFT to the rows.
    for row in x[..., 0]:
        _fft_inplace(row, scratch)

    # Transpose again before returning.
    transpose_square(x)


def _fft_inplace_oddpow(x, scratch):
    """In-place FFT of length 2^odd"""
    # This code is based on `_fft_inplace_evenpow`, but it has to
    # account for some additional complications.

    vec_len = x.shape[0]
    # `vec_len` is an odd power of 2, so we cannot reshape `x` to a
    # matrix square. Instead, we'll (conceptually) reshape it to a
    # matrix that's twice as wide as it is high. E.g., `[1 ... 8]`
    # becomes `[1 2 3 4]`
    #         `[5 6 7 8]`.
    col_len = 1 << (lb_exact(vec_len) >> 1)
    row_len = col_len << 1

    # We can only perform efficient, in-place transposes on square
    # matrices, so we will actually treat this as a square matrix of
    # 2-tuples, e.g. `[(1 2) (3 4)]`
    #                `[(5 6) (7 8)]`.
    # Note that we can currently `.reshape` it to our intended wide
    # matrix (although this is broken by transposition).
    x.shape = col_len, col_len, 2

    # We want to apply FFT to each column. We transpose our
    # matrix-of-tuples and get something like `[(1 2) (5 6)]`
    #                                         `[(3 4) (7 8)]`.
    # Note that each row of the transposed matrix represents two columns
    # of the original matrix. We can deinterleave the values to recover
    # the original columns.
    transpose_square(x)

    for i, row_pair in enumerate(x):
        # `row_pair` represents two columns of the original matrix.
        # Their values must be deinterleaved to recover the columns.
        row_pair.shape = row_len,
        _deinterleave(row_pair, scratch)
        # The below are rows of the transposed matrix(/cols of the
        # original matrix.
        row0 = row_pair[:col_len]
        row1 = row_pair[col_len:]

        # Apply FFT and twiddle factors to each.
        _fft_inplace(row0, scratch)
        for j in range(col_len):
            row0[j] *= np.exp(-2j * np.pi * ((2 * i) * j) / vec_len)
        _fft_inplace(row1, scratch)
        for j in range(col_len):
            row1[j] *= np.exp(-2j * np.pi * ((2 * i + 1) * j) / vec_len)

        # Re-interleave them and transpose back.
        _interleave(row_pair, scratch)

    transpose_square(x)

    # Recursively apply FFT to each row of the matrix.
    for row in x:
        # Turn vec of 2-tuples into vec of single elements.
        row.shape = row_len,
        _fft_inplace(row, scratch)

    # Transpose again before returning. This again involves
    # deinterleaving.
    transpose_square(x)
    for row_pair in x:
        row_pair.shape = row_len,
        _deinterleave(row_pair, scratch)


def _fft_inplace(x, scratch):
    """In-place FFT."""
    # Avoid modifying the shape of the original.
    # This does not copy the buffer.
    x = x.view()
    assert x.flags['C_CONTIGUOUS']
    
    n, = x.shape
    if n == 1:
        return
    if n == 2:
        x0, x1 = x
        x[0] = x0 + x1
        x[1] = x0 - x1
        return

    lb_n = lb_exact(n)    
    is_odd = lb_n & 1 != 0
    if is_odd:
        _fft_inplace_oddpow(x, scratch)
    else:
        _fft_inplace_evenpow(x, scratch)


def _scratch_length(lb_n):
    """Find the amount of scratch space required to run the FFT.

    Layers where the input's length is an even power of two do not
    require scratch space, but the layers where that power is odd do.
    """
    if lb_n == 0:
        # Length-1 input.
        return 0
    # Repeatedly halve lb_n as long as it's even. This is the same as
    # `n = sqrt(n)`, where the `sqrt` is exact.
    while lb_n & 1 == 0:
        lb_n >>= 1
    # `lb_n` is now odd, so `n` is not an even power of 2.
    lb_res = (lb_n - 1) >> 1
    if lb_res == 0:
        # Special case (n == 2 or n == 4): no scratch needed.
        return 0
    return 1 << lb_res


def fft(x):
    """Returns the FFT of `x`.

    This is a wrapper around an in-place routine, provided for user
    convenience.
    """
    n, = x.shape
    lb_n = lb_exact(n)  # Raises if not a power of 2.
    # We have one scratch buffer for the whole algorithm. If we were to
    # parallelize it, we'd need one thread-local buffer for each worker
    # thread.
    scratch_len = _scratch_length(lb_n)
    if scratch_len == 0:
        scratch = None
    else:
        scratch = np.empty_like(x, shape=scratch_len, order='C', subok=False)

    res = x.copy(order='C')
    _fft_inplace(res, scratch)

    return res


if __name__ == "__main__":
    LENGTH = 1 << 10
    v = np.random.normal(size=LENGTH).astype(complex)
    print(v)
    numpy_fft = np.fft.fft(v)
    print(numpy_fft)
    our_fft = fft(v)
    print(our_fft)
    print(np.isclose(numpy_fft, our_fft).all())
