#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


def numerical_gradient(W, dx):
    """
    Compute numerical gradient of the wavefront.

    np.gradient returns:
        dW/daxis0, dW/daxis1

    Since axis0 corresponds to y and axis1 corresponds to x:

        dW_dy, dW_dx = np.gradient(...)
    """
    W0 = np.nan_to_num(W, nan=0.0)
    dW_dy, dW_dx = np.gradient(W0, dx, dx)
    return dW_dx, dW_dy


# In[ ]:


def subaperture_masks(X, Y, pupil_mask, n_lenslets=12, min_fill=0.5):
    """
    Divide the pupil into square Shack-Hartmann sub-apertures.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate grids.
    pupil_mask : ndarray
        Circular pupil mask.
    n_lenslets : int
        Number of lenslets across the full grid.
    min_fill : float
        Minimum fraction of a sub-aperture that must lie inside the pupil.

    Returns
    -------
    centers : ndarray
        Sub-aperture center coordinates.
    masks : list of ndarray
        Boolean masks for valid sub-apertures.
    """
    x_min, x_max = np.nanmin(X), np.nanmax(X)
    y_min, y_max = np.nanmin(Y), np.nanmax(Y)

    x_edges = np.linspace(x_min, x_max, n_lenslets + 1)
    y_edges = np.linspace(y_min, y_max, n_lenslets + 1)

    masks = []
    centers = []

    for ix in range(n_lenslets):
        for iy in range(n_lenslets):
            cell = (
                (X >= x_edges[ix])
                & (X < x_edges[ix + 1])
                & (Y >= y_edges[iy])
                & (Y < y_edges[iy + 1])
            )

            valid = cell & pupil_mask
            fill = valid.sum() / max(cell.sum(), 1)

            if fill >= min_fill:
                masks.append(valid)
                centers.append(
                    [
                        0.5 * (x_edges[ix] + x_edges[ix + 1]),
                        0.5 * (y_edges[iy] + y_edges[iy + 1]),
                    ]
                )

    return np.array(centers), masks


# In[ ]:


def measure_slopes(
    W,
    pupil_mask,
    X,
    Y,
    n_lenslets=12,
    min_fill=0.5,
    noise_std=0.0,
    seed=1,
):
    """
    Measure mean local wavefront slopes in every valid sub-aperture.

    Parameters
    ----------
    W : ndarray
        Wavefront map. Unit can be phase, OPD, or arbitrary wavefront unit.
    pupil_mask : ndarray
        Pupil mask.
    X, Y : ndarray
        Coordinate grids.
    n_lenslets : int
        Number of lenslets across the full pupil grid.
    min_fill : float
        Minimum valid area fraction per sub-aperture.
    noise_std : float
        Standard deviation of Gaussian slope noise.
    seed : int
        Random seed.

    Returns
    -------
    centers : ndarray
        Valid sub-aperture center coordinates.
    slopes : ndarray
        Measured slopes, shape = (N_subapertures, 2).

    Notes
    -----
    This is a geometric Shack-Hartmann model:

        sx = <dW/dx>
        sy = <dW/dy>

    A more realistic model would simulate diffraction spots and centroiding.
    """
    dx = X[0, 1] - X[0, 0]
    dWdx, dWdy = numerical_gradient(W, dx)

    centers, masks = subaperture_masks(
        X,
        Y,
        pupil_mask,
        n_lenslets=n_lenslets,
        min_fill=min_fill,
    )

    slopes = []

    for m in masks:
        sx = np.mean(dWdx[m])
        sy = np.mean(dWdy[m])
        slopes.append([sx, sy])

    slopes = np.array(slopes)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        slopes = slopes + rng.normal(scale=noise_std, size=slopes.shape)

    return centers, slopes


# In[ ]:


def build_response_matrix(
    modes,
    pupil_mask,
    X,
    Y,
    n_lenslets=12,
    min_fill=0.5,
):
    """
    Build the Shack-Hartmann response matrix.

    The modal wavefront is written as:

        W(x, y) = sum_j a_j Z_j(x, y)

    The Shack-Hartmann slope vector is:

        s = A a + n

    where:
        s : measured slope vector
        A : response matrix
        a : modal coefficient vector
        n : measurement noise

    Each column of A is the slope signal caused by one basis mode.
    """
    names = list(modes.keys())
    columns = []
    reference_centers = None

    for name in names:
        centers, slopes = measure_slopes(
            modes[name],
            pupil_mask,
            X,
            Y,
            n_lenslets=n_lenslets,
            min_fill=min_fill,
            noise_std=0.0,
        )

        if reference_centers is None:
            reference_centers = centers
        else:
            if centers.shape != reference_centers.shape or not np.allclose(
                centers, reference_centers
            ):
                raise RuntimeError(
                    "Sub-aperture geometry changed between modes."
                )

        columns.append(slopes.reshape(-1))

    A = np.column_stack(columns)

    return A, names, reference_centers


# In[ ]:


def reconstruct_modal_coefficients(
    slopes,
    response_matrix,
    rcond=1e-4,
):
    """
    Least-squares modal reconstruction.

    Solve:

        coeffs = argmin ||A coeffs - slopes||^2

    Parameters
    ----------
    slopes : ndarray
        Measured Shack-Hartmann slopes.
    response_matrix : ndarray
        Response matrix A.
    rcond : float
        Cutoff for small singular values.

    Returns
    -------
    coeffs : ndarray
        Estimated modal coefficients.
    residuals : ndarray
        Least-squares residuals from numpy.
    rank : int
        Rank of the response matrix.
    singular_values : ndarray
        Singular values of the response matrix.
    """
    s = slopes.reshape(-1)

    coeffs, residuals, rank, singular_values = np.linalg.lstsq(
        response_matrix,
        s,
        rcond=rcond,
    )

    return coeffs, residuals, rank, singular_values


# In[ ]:


def remove_piston(W, mask):
    """
    Remove piston inside the pupil.
    """
    out = np.array(W, dtype=float, copy=True)
    out[mask] -= np.nanmean(out[mask])
    out[~mask] = np.nan
    return out


def synthesize_from_coefficients(
    coeffs,
    modes,
    names,
    pupil_mask,
    remove_mean=True,
):
    """
    Reconstruct a wavefront map from modal coefficients.
    """
    W = np.zeros_like(next(iter(modes.values())), dtype=float)

    for c, name in zip(coeffs, names):
        W += c * np.nan_to_num(modes[name], nan=0.0)

    W = np.where(pupil_mask, W, np.nan)

    if remove_mean:
        W = remove_piston(W, pupil_mask)

    return W


# In[ ]:


def reconstruct_wavefront(
    slopes,
    response_matrix,
    modes,
    names,
    pupil_mask,
    rcond=1e-4,
):
    """
    Full modal wavefront reconstruction from Shack-Hartmann slopes.
    """
    coeffs, residuals, rank, singular_values = reconstruct_modal_coefficients(
        slopes,
        response_matrix,
        rcond=rcond,
    )

    W_rec = synthesize_from_coefficients(
        coeffs,
        modes,
        names,
        pupil_mask,
        remove_mean=True,
    )

    return coeffs, W_rec, residuals, rank, singular_values


def rms(W, mask):
    """
    RMS inside the pupil after mean subtraction.
    """
    vals = np.asarray(W)[mask]
    vals = vals[np.isfinite(vals)]
    vals = vals - np.mean(vals)
    return float(np.sqrt(np.mean(vals**2)))


def residual_wavefront(W_true, W_rec, mask):
    """
    Compute residual wavefront after removing piston.
    """
    res = np.asarray(W_true) - np.asarray(W_rec)
    return remove_piston(res, mask)

