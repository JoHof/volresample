import numpy as np
from numba import njit, prange

_INF = 1e20

@njit(fastmath=True)
def _distance_transform_1d(f, n, spacing):
    """
    1D squared distance transform (Felzenszwalb & Huttenlocher).
    f: 1D array of floats (costs: 0 at feature positions, large at others)
    n: length
    spacing: voxel spacing along this axis (float)
    returns array of squared distances (in physical units)
    """
    v = np.empty(n, np.int64)          # locations of parabolas in lower envelope
    z = np.empty(n + 1, np.float64)    # locations of boundaries between parabolas
    d = np.empty(n, np.float64)        # output (squared distances)

    v[0] = 0
    z[0] = -_INF
    z[1] = _INF
    k = 0
    s2 = spacing * spacing

    for q in range(1, n):
        # compute intersection
        # handle potential division by zero (q - v[k] >= 1 because q>v[k])
        num = (f[q] + (q * q) * s2) - (f[v[k]] + (v[k] * v[k]) * s2)
        den = 2.0 * s2 * (q - v[k])
        if den == 0.0:
            s = _INF
        else:
            s = num / den

        while s <= z[k]:
            k -= 1
            num = (f[q] + (q * q) * s2) - (f[v[k]] + (v[k] * v[k]) * s2)
            den = 2.0 * s2 * (q - v[k])
            if den == 0.0:
                s = _INF
            else:
                s = num / den

        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = _INF

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        d[q] = dq * dq * s2 + f[v[k]]

    return d


@njit(parallel=True, fastmath=True)
def edt_3d_numba(matrix, voxelspacing):
    """
    3D Euclidean Distance Transform (Numba accelerated).

    Parameters
    ----------
    matrix : ndarray
        3D array (boolean or numeric). Semantics:
        Non-zero voxels are considered 'foreground' and distances are computed
        to the nearest zero voxel (same as scipy.ndimage.distance_transform_edt).
    voxelspacing : sequence of 3 floats
        Physical spacing for (z, y, x) axes (sz, sy, sx).

    Returns
    -------
    distances : ndarray of floats
        3D array of Euclidean distances (same shape as input), in physical units.
    """
    # ensure inputs are correct shape
    zdim, ydim, xdim = matrix.shape
    sz, sy, sx = voxelspacing[0], voxelspacing[1], voxelspacing[2]

    # Step 0: build initial f where f = 0 at zeros (background), INF elsewhere.
    # This follows scipy semantics: compute distances for non-zero voxels to nearest zero.
    f = np.empty((zdim, ydim, xdim), dtype=np.float64)
    for i in prange(zdim):
        for j in range(ydim):
            for k in range(xdim):
                if matrix[i, j, k] == 0:
                    f[i, j, k] = 0.0
                else:
                    f[i, j, k] = _INF

    # Temporary arrays for intermediate squared-distance results
    tmp = np.empty_like(f)   # after x-axis pass
    tmp2 = np.empty_like(f)  # after y-axis pass

    # Pass 1: transform along x (last axis) for each (z,y) line
    for i in prange(zdim):
        for j in range(ydim):
            # extract 1d array
            line = f[i, j, :]
            dline = _distance_transform_1d(line, xdim, sx)
            for k in range(xdim):
                tmp[i, j, k] = dline[k]

    # Pass 2: transform along y for each (z,x) line
    for i in prange(zdim):
        for k in range(xdim):
            # build 1d array of length ydim
            work = np.empty(ydim, np.float64)
            for j in range(ydim):
                work[j] = tmp[i, j, k]
            dline = _distance_transform_1d(work, ydim, sy)
            for j in range(ydim):
                tmp2[i, j, k] = dline[j]

    # Pass 3: transform along z for each (y,x) line -> final squared distances
    out_sq = np.empty_like(f)
    for j in prange(ydim):
        for k in range(xdim):
            work = np.empty(zdim, np.float64)
            for i in range(zdim):
                work[i] = tmp2[i, j, k]
            dline = _distance_transform_1d(work, zdim, sz)
            for i in range(zdim):
                out_sq[i, j, k] = dline[i]

    # Convert squared distances to distances (sqrt)
    out = np.empty_like(out_sq)
    for i in prange(zdim):
        for j in range(ydim):
            for k in range(xdim):
                val = out_sq[i, j, k]
                if val < 0.0:
                    # numerical safety
                    val = 0.0
                out[i, j, k] = np.sqrt(val)

    return out
