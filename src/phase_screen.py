#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


def circular_mask_from_grid(X, Y, diameter):
    """
    Create circular pupil mask from coordinate grids.
    """
    R = np.sqrt(X**2 + Y**2)
    return R <= diameter / 2


def remove_piston(screen, mask):
    """
    Remove piston inside the aperture.
    """
    out = np.array(screen, dtype=float, copy=True)
    out[mask] -= np.mean(out[mask])
    out[~mask] = np.nan
    return out


def rms(screen, mask):
    """
    RMS inside the pupil after mean removal.
    """
    vals = np.asarray(screen)[mask]
    vals = vals[np.isfinite(vals)]
    vals = vals - np.mean(vals)
    return float(np.sqrt(np.mean(vals**2)))


# In[ ]:


def fourier_phase_screen(
    N=256,
    delta=0.01,
    r0=0.15,
    L0=25.0,
    diameter=1.0,
    wavelength=500e-9,
    seed=1,
    target_rms_rad=None,
):
    """
    Generate a demo von Karman / Kolmogorov-like atmospheric phase screen.

    Parameters
    ----------
    N : int
        Grid size.
    delta : float
        Grid spacing in meters.
    r0 : float
        Fried parameter in meters.
    L0 : float
        Outer scale in meters. Use np.inf for pure Kolmogorov-like spectrum.
    diameter : float
        Pupil diameter in meters.
    wavelength : float
        Wavelength in meters. Kept for reference.
    seed : int
        Random seed.
    target_rms_rad : float or None
        Target RMS phase in radians inside the pupil.
        If None, use approximate scaling sqrt(1.03 * (D/r0)^(5/3)).

    Returns
    -------
    phase : ndarray
        Atmospheric phase screen in radians.
    X, Y : ndarray
        Coordinate grids in meters.
    mask : ndarray
        Circular pupil mask.

    Notes
    -----
    The spectral shape follows the usual atmospheric PSD form:

        PSD_phi(f) ~ 0.023 r0^(-5/3) (f^2 + f0^2)^(-11/6)

    The final RMS is explicitly normalized to make this robust as a learning
    simulator. Therefore this is not a calibrated AO performance simulator.
    """
    rng = np.random.default_rng(seed)

    fx = np.fft.fftfreq(N, d=delta)
    fy = np.fft.fftfreq(N, d=delta)
    FX, FY = np.meshgrid(fx, fy)
    f = np.sqrt(FX**2 + FY**2)

    if np.isinf(L0) or L0 is None:
        f0 = 0.0
    else:
        f0 = 1.0 / L0

    psd = 0.023 * r0 ** (-5.0 / 3.0) * (f**2 + f0**2) ** (-11.0 / 6.0)
    psd[0, 0] = 0.0

    random_complex = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    fourier_coeff = random_complex * np.sqrt(psd)

    phase = np.fft.ifft2(fourier_coeff).real

    x = (np.arange(N) - N // 2) * delta
    X, Y = np.meshgrid(x, x)

    mask = circular_mask_from_grid(X, Y, diameter)

    phase = phase - np.mean(phase[mask])

    if target_rms_rad is None:
        target_rms_rad = np.sqrt(1.03 * (diameter / r0) ** (5.0 / 3.0))

    current = np.std(phase[mask])

    if current > 0:
        phase *= target_rms_rad / current

    phase = np.where(mask, phase, np.nan)

    return phase, X, Y, mask


# In[ ]:


def phase_to_opd(phase_rad, wavelength):
    """
    Convert phase in radians to optical path difference in meters.

        phase = 2*pi*OPD/lambda
    """
    return phase_rad * wavelength / (2 * np.pi)


def opd_to_phase(opd_m, wavelength):
    """
    Convert optical path difference in meters to phase in radians.
    """
    return 2 * np.pi * opd_m / wavelength


def frozen_flow_shift(screen, shift_x_pix=0, shift_y_pix=0):
    """
    Shift a phase screen with periodic boundary conditions.

    This is useful for a toy frozen-flow atmosphere model.
    """
    return np.roll(
        np.roll(screen, shift_y_pix, axis=0),
        shift_x_pix,
        axis=1,
    )


# In[ ]:





# In[ ]:




