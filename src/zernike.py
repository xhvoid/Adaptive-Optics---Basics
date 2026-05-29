#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


def make_pupil_grid(N=256, diameter=1.0):
    """
    Create a circular pupil grid.

    Parameters
    ----------
    N : int
        Grid size.
    diameter : float
        Physical pupil diameter. Unit can be arbitrary, but must be consistent.

    Returns
    -------
    X, Y : ndarray
        Cartesian coordinate grids.
    rho : ndarray
        Normalized radius, rho = r / (D/2).
    theta : ndarray
        Azimuthal angle.
    mask : ndarray
        Circular pupil mask.
    dx : float
        Grid spacing.
    """
    x = np.linspace(-diameter / 2, diameter / 2, N)
    X, Y = np.meshgrid(x, x)

    R = np.sqrt(X**2 + Y**2)
    rho = R / (diameter / 2)
    theta = np.arctan2(Y, X)

    mask = rho <= 1.0
    dx = x[1] - x[0]

    return X, Y, rho, theta, mask, dx


def remove_piston(W, mask):
    """
    Remove piston, i.e. subtract the mean value inside the pupil.
    """
    W2 = np.array(W, dtype=float, copy=True)
    W2[mask] -= np.nanmean(W2[mask])
    W2[~mask] = np.nan
    return W2


def rms(W, mask):
    """
    RMS of a wavefront inside the pupil after piston removal.
    """
    vals = np.asarray(W)[mask]
    vals = vals[np.isfinite(vals)]
    vals = vals - np.mean(vals)
    return float(np.sqrt(np.mean(vals**2)))


def zernike_named_modes(rho, theta, mask, include_piston=False, normalized=True):
    """
    Generate a small set of low-order Zernike-like modes.

    This is enough for a minimal Shack-Hartmann WFS simulator.

    Modes included:
        piston
        tip_x
        tip_y
        defocus
        astig_45
        astig_0
        coma_x
        coma_y
        trefoil_x
        trefoil_y
        spherical

    Notes
    -----
    These are practical low-order Zernike expressions.
    For a learning simulator this is sufficient.
    For high-precision AO modeling, use a rigorously normalized Zernike package.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    r2 = rho**2

    if normalized:
        modes = {
            "piston": np.ones_like(rho),
            "tip_x": 2 * x,
            "tip_y": 2 * y,
            "defocus": np.sqrt(3) * (2 * r2 - 1),
            "astig_45": np.sqrt(6) * 2 * x * y,
            "astig_0": np.sqrt(6) * (x**2 - y**2),
            "coma_x": np.sqrt(8) * (3 * r2 - 2) * x,
            "coma_y": np.sqrt(8) * (3 * r2 - 2) * y,
            "trefoil_x": np.sqrt(8) * (x**3 - 3 * x * y**2),
            "trefoil_y": np.sqrt(8) * (3 * x**2 * y - y**3),
            "spherical": np.sqrt(5) * (6 * r2**2 - 6 * r2 + 1),
        }
    else:
        modes = {
            "piston": np.ones_like(rho),
            "tip_x": x,
            "tip_y": y,
            "defocus": 2 * r2 - 1,
            "astig_45": 2 * x * y,
            "astig_0": x**2 - y**2,
            "coma_x": (3 * r2 - 2) * x,
            "coma_y": (3 * r2 - 2) * y,
            "trefoil_x": x**3 - 3 * x * y**2,
            "trefoil_y": 3 * x**2 * y - y**3,
            "spherical": 6 * r2**2 - 6 * r2 + 1,
        }

    if not include_piston:
        modes.pop("piston")

    for key in list(modes.keys()):
        arr = np.asarray(modes[key], dtype=float)
        arr = np.where(mask, arr, np.nan)
        modes[key] = arr

    return modes


def synthesize_wavefront(modes, coeffs, mask, remove_mean=True):
    """
    Synthesize a wavefront from modal coefficients.

    Parameters
    ----------
    modes : dict
        Dictionary of Zernike-like basis modes.
    coeffs : dict
        Dictionary of modal coefficients.
    mask : ndarray
        Pupil mask.
    remove_mean : bool
        Whether to remove piston after synthesis.

    Returns
    -------
    W : ndarray
        Synthesized wavefront.
    """
    first_mode = next(iter(modes.values()))
    W = np.zeros_like(first_mode, dtype=float)

    for name, coeff in coeffs.items():
        if name not in modes:
            raise KeyError(
                f"Mode '{name}' is not available. "
                f"Available modes: {list(modes)}"
            )

        W += coeff * np.nan_to_num(modes[name], nan=0.0)

    W = np.where(mask, W, np.nan)

    if remove_mean:
        W = remove_piston(W, mask)

    return W


# In[ ]:


import math
import numpy as np


def zernike_radial(n, m, rho):
    """
    Radial part of the Zernike polynomial R_n^m(rho).

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal order. Use m >= 0.
    rho : ndarray
        Normalized radial coordinate, 0 <= rho <= 1.

    Returns
    -------
    R : ndarray
        Radial polynomial R_n^m(rho).

    Notes
    -----
    Zernike modes only exist when:
        n >= m
        n - m is even
    """
    m = abs(m)

    if n < m:
        raise ValueError("Zernike mode requires n >= |m|.")

    if (n - m) % 2 != 0:
        raise ValueError("Zernike mode requires n - |m| to be even.")

    R = np.zeros_like(rho, dtype=float)

    max_k = (n - m) // 2

    for k in range(max_k + 1):
        numerator = (-1)**k * math.factorial(n - k)
        denominator = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )

        R += numerator / denominator * rho ** (n - 2 * k)

    return R


def zernike_nm(n, m, rho, theta, mask, normalization=True):
    """
    Generate one real-valued Zernike mode.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal order.

        m = 0:
            axisymmetric mode

        m > 0:
            cosine mode

        m < 0:
            sine mode

    rho : ndarray
        Normalized radial coordinate.
    theta : ndarray
        Azimuthal coordinate.
    mask : ndarray
        Pupil mask.
    normalization : bool
        If True, use standard orthonormal normalization on the unit disk.

    Returns
    -------
    Z : ndarray
        Zernike mode on the pupil.
    """
    abs_m = abs(m)

    R = zernike_radial(n, abs_m, rho)

    if m == 0:
        Z = R
        if normalization:
            Z *= np.sqrt(n + 1)

    elif m > 0:
        Z = R * np.cos(abs_m * theta)
        if normalization:
            Z *= np.sqrt(2 * (n + 1))

    else:
        Z = R * np.sin(abs_m * theta)
        if normalization:
            Z *= np.sqrt(2 * (n + 1))

    Z = np.where(mask, Z, np.nan)

    return Z


def generate_zernike_modes(
    rho,
    theta,
    mask,
    max_radial_order=6,
    include_piston=False,
    normalization=True,
):
    """
    Generate real-valued Zernike modes up to a given radial order.

    Parameters
    ----------
    rho, theta : ndarray
        Normalized polar pupil coordinates.
    mask : ndarray
        Pupil mask.
    max_radial_order : int
        Maximum radial order n.
    include_piston : bool
        Whether to include Z_0^0.
        For Shack-Hartmann WFS reconstruction, piston is usually excluded.
    normalization : bool
        Whether to use standard orthonormal normalization.

    Returns
    -------
    modes : dict
        Dictionary of Zernike modes.

    Naming convention
    -----------------
    Zernike modes are named as:

        Z{n}_{m:+d}

    Examples:
        Z2_+0  : defocus-like mode
        Z2_+2  : astigmatism cosine mode
        Z2_-2  : astigmatism sine mode
        Z3_+1  : coma-like cosine mode
        Z3_-1  : coma-like sine mode

    Notes
    -----
    This is not Noll indexing. It uses direct (n, m) indexing.
    """
    modes = {}

    for n in range(max_radial_order + 1):
        for m_abs in range(n + 1):
            if (n - m_abs) % 2 != 0:
                continue

            if n == 0 and m_abs == 0:
                if include_piston:
                    name = "Z0_+0"
                    modes[name] = zernike_nm(
                        n,
                        0,
                        rho,
                        theta,
                        mask,
                        normalization=normalization,
                    )
                continue

            if m_abs == 0:
                name = f"Z{n}_+0"
                modes[name] = zernike_nm(
                    n,
                    0,
                    rho,
                    theta,
                    mask,
                    normalization=normalization,
                )
            else:
                name_cos = f"Z{n}_+{m_abs}"
                name_sin = f"Z{n}_-{m_abs}"

                modes[name_cos] = zernike_nm(
                    n,
                    +m_abs,
                    rho,
                    theta,
                    mask,
                    normalization=normalization,
                )

                modes[name_sin] = zernike_nm(
                    n,
                    -m_abs,
                    rho,
                    theta,
                    mask,
                    normalization=normalization,
                )

    return modes


def number_of_zernike_modes(max_radial_order, include_piston=False):
    """
    Number of Zernike modes up to radial order n.

    Total number including piston:
        (n + 1)(n + 2) / 2
    """
    total = (max_radial_order + 1) * (max_radial_order + 2) // 2

    if not include_piston:
        total -= 1

    return total


# In[ ]:




