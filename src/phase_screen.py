"""
Atmospheric phase-screen utilities for compact AO simulations.

The generator below draws a von Karman phase screen in the Fourier domain and
adds low-frequency subharmonics (Lane et al. 1992), so the screen keeps the
tip/tilt power that a plain FFT screen misses. Without normalization, the
screen has the absolute strength set by r0 and L0.

``von_karman_phase_variance`` gives the expected aperture variance for the
same spectrum, which is how the generator is checked in the tests.
Normalizing to a target RMS is still available for controlled tests.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1, jv


def circular_mask_from_grid(X: np.ndarray, Y: np.ndarray, diameter: float) -> np.ndarray:
    """Create a circular pupil mask from coordinate grids."""
    R = np.sqrt(X**2 + Y**2)
    return R <= diameter / 2.0


def pupil_grid(
    N: int,
    delta: float,
    diameter: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Coordinate grids and pupil mask on the phase-screen grid convention.

    The origin sits at pixel ``N // 2``. ``diameter=None`` returns a mask that
    covers the whole square grid.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    x = (np.arange(N) - N // 2) * delta
    X, Y = np.meshgrid(x, x)
    if diameter is None:
        mask = np.ones((N, N), dtype=bool)
    else:
        if diameter <= 0:
            raise ValueError("diameter must be positive.")
        mask = circular_mask_from_grid(X, Y, diameter)
    return X, Y, mask


def remove_piston(screen: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove piston inside the aperture and set values outside to NaN."""
    out = np.array(screen, dtype=float, copy=True)
    finite = mask & np.isfinite(out)
    if np.any(finite):
        out[finite] -= np.mean(out[finite])
    out[~mask] = np.nan
    return out


def rms(screen: np.ndarray, mask: np.ndarray) -> float:
    """RMS inside the pupil after mean removal."""
    vals = np.asarray(screen, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    vals = vals - np.mean(vals)
    return float(np.sqrt(np.mean(vals**2)))


def r0_from_seeing(seeing_arcsec: float, wavelength: float = 500e-9) -> float:
    """
    Convert seeing FWHM to Fried parameter r0.

    Parameters
    ----------
    seeing_arcsec:
        Seeing FWHM in arcseconds.
    wavelength:
        Wavelength in meters at which the seeing is quoted.
    """
    if seeing_arcsec <= 0:
        raise ValueError("seeing_arcsec must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    seeing_rad = seeing_arcsec / 206265.0
    return float(0.98 * wavelength / seeing_rad)


def scale_r0_with_wavelength(
    r0_ref: float,
    wavelength: float,
    wavelength_ref: float = 500e-9,
) -> float:
    """
    Scale Fried parameter with wavelength.

        r0(lambda) = r0(lambda_ref) * (lambda/lambda_ref)^(6/5)
    """
    if r0_ref <= 0:
        raise ValueError("r0_ref must be positive.")
    if wavelength <= 0 or wavelength_ref <= 0:
        raise ValueError("wavelength and wavelength_ref must be positive.")

    return float(r0_ref * (wavelength / wavelength_ref) ** (6.0 / 5.0))


def _outer_scale_frequency(L0: float | None) -> float:
    if L0 is None or np.isinf(L0):
        return 0.0
    if L0 <= 0:
        raise ValueError("L0 must be positive, np.inf, or None.")
    return 1.0 / L0


def von_karman_phase_psd(f: np.ndarray, r0: float, L0: float | None = None) -> np.ndarray:
    """
    Von Karman phase power spectral density in rad^2 m^2.

        PSD(f) = 0.023 r0^(-5/3) (f^2 + 1/L0^2)^(-11/6)

    ``f`` is the spatial frequency in cycles per metre. ``L0=None`` gives the
    Kolmogorov spectrum, which diverges at f = 0.
    """
    if r0 <= 0:
        raise ValueError("r0 must be positive.")
    f0 = _outer_scale_frequency(L0)
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        return 0.023 * r0 ** (-5.0 / 3.0) * (f**2 + f0**2) ** (-11.0 / 6.0)


def von_karman_phase_variance(
    diameter: float,
    r0: float,
    L0: float | None = None,
    remove_tilt: bool = False,
) -> float:
    """
    Expected phase variance in rad^2 over a circular aperture.

    The PSD is integrated against the filter that removes piston (and
    optionally tip/tilt) over the aperture, with x = pi D f:

        piston filter = (2 J1(x) / x)^2
        tilt filter   = 16 (J2(x) / x)^2

    For a Kolmogorov spectrum this gives Noll's 1.0299 (D/r0)^(5/3) with
    piston removed and 0.134 (D/r0)^(5/3) with tip/tilt also removed.
    """
    if diameter <= 0:
        raise ValueError("diameter must be positive.")

    # Integrate in log f; the filter makes the integrand vanish at both ends.
    f = np.logspace(-7.0, 4.0, 40001) / diameter
    x = np.pi * diameter * f
    keep = 1.0 - (2.0 * j1(x) / x) ** 2
    if remove_tilt:
        keep -= 16.0 * (jv(2, x) / x) ** 2
    integrand = von_karman_phase_psd(f, r0, L0) * keep * 2.0 * np.pi * f
    y = integrand * f
    u = np.log(f)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(u)))


# FFT cells with |kx|, |ky| <= _LOW_BLOCK are drawn by explicit summation
# instead of the FFT, because the spectrum changes steeply across them.
_LOW_BLOCK = 2
_SUBCELLS = 4


def _low_frequency_phase(
    x: np.ndarray,
    screen_size_m: float,
    r0: float,
    L0: float | None,
    rng: np.random.Generator,
    n_levels: int,
) -> np.ndarray:
    """
    Phase from the lowest spatial frequencies, summed explicitly.

    Level 0 covers the FFT cells with |kx|, |ky| <= 2 (cell size 1/S, where S
    is the screen size), except the zero-frequency cell. Each subharmonic
    level p splits the remaining central cell into 3 x 3 cells of size
    1 / (3^p S) and keeps the outer eight (Lane et al. 1992).

    Every cell is split into 4 x 4 subcells. Each subcell gets an independent
    complex Gaussian coefficient with variance PSD(f) * area at its own
    centre frequency, so the steep spectrum is integrated over the cell
    rather than sampled once at its centre.
    """
    freqs = []
    weights = []
    sub = (np.arange(_SUBCELLS) + 0.5) / _SUBCELLS - 0.5

    def add_cell(cx: float, cy: float, size: float) -> None:
        FX, FY = np.meshgrid(cx + sub * size, cy + sub * size)
        area = (size / _SUBCELLS) ** 2
        freqs.append(np.column_stack([FX.ravel(), FY.ravel()]))
        weights.append(np.sqrt(von_karman_phase_psd(np.hypot(FX, FY), r0, L0).ravel() * area))

    df0 = 1.0 / screen_size_m
    for kx in range(-_LOW_BLOCK, _LOW_BLOCK + 1):
        for ky in range(-_LOW_BLOCK, _LOW_BLOCK + 1):
            if kx or ky:
                add_cell(kx * df0, ky * df0, df0)

    for p in range(1, n_levels + 1):
        df = df0 / 3.0**p
        for kx in (-1, 0, 1):
            for ky in (-1, 0, 1):
                if kx or ky:
                    add_cell(kx * df, ky * df, df)

    f = np.concatenate(freqs)
    w = np.concatenate(weights)
    c = (rng.normal(size=w.size) + 1j * rng.normal(size=w.size)) * w

    # exp(2 pi i (fx x + fy y)) separates into an x factor and a y factor,
    # so the sum over frequencies is one matrix product.
    Ex = np.exp(2j * np.pi * np.outer(x, f[:, 0]))
    Ey = np.exp(2j * np.pi * np.outer(x, f[:, 1]))
    low = ((Ey * c) @ Ex.T).real
    return low - np.mean(low)


def _default_subharmonic_levels(screen_size_m: float, L0: float | None) -> int:
    # Smallest p >= 3 with 1 / (3^p S) <= 1 / (3 L0), capped at 12.
    if _outer_scale_frequency(L0) == 0.0:
        return 12
    p = int(np.ceil(np.log(3.0 * L0 / screen_size_m) / np.log(3.0)))
    return int(np.clip(p, 3, 12))


def fourier_phase_screen(
    N: int = 256,
    delta: float = 0.01,
    r0: float = 0.15,
    L0: float | None = 25.0,
    diameter: float | None = 1.0,
    wavelength: float = 500e-9,
    seed: int | None = 1,
    target_rms_rad: float | None = None,
    normalize_rms: bool | None = None,
    subharmonics: bool = True,
    n_subharmonic_levels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a von Karman atmospheric phase screen.

    Parameters
    ----------
    N:
        Grid size.
    delta:
        Grid spacing in meters.
    r0:
        Fried parameter in meters, at the phase-screen wavelength.
    L0:
        Outer scale in meters. Use ``np.inf`` or ``None`` for a Kolmogorov
        spectrum.
    diameter:
        Pupil diameter in meters. ``None`` keeps the whole square grid, which
        is what a frozen-flow sheet needs (see ``frozen_flow_shift``).
    wavelength:
        Wavelength in meters. Kept for parameter bookkeeping; the returned
        screen is phase in radians at this wavelength.
    seed:
        Random seed.
    target_rms_rad:
        Target RMS phase in radians inside the pupil.
    normalize_rms:
        Rescale the screen to a target pupil RMS. The default (``None``)
        normalizes only when ``target_rms_rad`` is given. ``True`` without a
        target uses the expected von Karman RMS for this aperture.
    subharmonics:
        Add the low-frequency correction. A plain FFT screen is periodic over
        N * delta and misses most of the tip/tilt power.
    n_subharmonic_levels:
        Number of subharmonic levels. The default reaches frequencies a few
        times below 1/L0 (at least 3 levels); a Kolmogorov spectrum, whose
        power keeps growing at large scales, gets 12.

    Returns
    -------
    phase:
        Atmospheric phase screen in radians, piston removed inside the mask,
        with NaN outside it.
    X, Y:
        Coordinate grids in meters.
    mask:
        Circular pupil mask (all True when ``diameter=None``).

    Notes
    -----
    The phase is the real part of a complex Gaussian field with the von
    Karman PSD (see ``von_karman_phase_psd``):

        phase(x) = Re sum_k c_k sqrt(PSD(f_k)) df exp(2 pi i f_k . x)

    with df = 1 / (N delta) and unit-variance real and imaginary parts in
    c_k. Its variance is therefore sum_k PSD(f_k) df^2, the discrete form of
    the integral of the PSD.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if r0 <= 0:
        raise ValueError("r0 must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    rng = np.random.default_rng(seed)
    X, Y, mask = pupil_grid(N, delta, diameter)

    df = 1.0 / (N * delta)
    fx = np.fft.fftfreq(N, d=delta)
    FX, FY = np.meshgrid(fx, fx)
    psd = von_karman_phase_psd(np.hypot(FX, FY), r0, L0)
    psd[0, 0] = 0.0
    if subharmonics:
        # These cells are drawn more accurately in _low_frequency_phase.
        k = np.rint(np.fft.fftfreq(N) * N)
        KX, KY = np.meshgrid(k, k)
        psd[(np.abs(KX) <= _LOW_BLOCK) & (np.abs(KY) <= _LOW_BLOCK)] = 0.0

    cn = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * np.sqrt(psd) * df
    # numpy's ifft2 divides by N^2; multiply back to get the plain sum above.
    phase = np.fft.ifft2(cn).real * N**2

    if subharmonics:
        if n_subharmonic_levels is None:
            n_subharmonic_levels = _default_subharmonic_levels(N * delta, L0)
        phase = phase + _low_frequency_phase(
            X[0], N * delta, r0, L0, rng, n_subharmonic_levels
        )

    phase = remove_piston(phase, mask)

    if normalize_rms is None:
        normalize_rms = target_rms_rad is not None
    if normalize_rms:
        if target_rms_rad is None:
            pupil_d = N * delta if diameter is None else diameter
            target_rms_rad = np.sqrt(von_karman_phase_variance(pupil_d, r0, L0))
        if target_rms_rad < 0:
            raise ValueError("target_rms_rad must be non-negative.")

        current = rms(phase, mask)
        if np.isfinite(current) and current > 0:
            phase = phase * (target_rms_rad / current)
            phase = remove_piston(phase, mask)

    return phase, X, Y, mask


def phase_to_opd(phase_rad: np.ndarray, wavelength: float) -> np.ndarray:
    """Convert phase in radians to optical path difference in meters."""
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    return np.asarray(phase_rad, dtype=float) * wavelength / (2.0 * np.pi)


def opd_to_phase(opd_m: np.ndarray, wavelength: float) -> np.ndarray:
    """Convert optical path difference in meters to phase in radians."""
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    return 2.0 * np.pi * np.asarray(opd_m, dtype=float) / wavelength


def frozen_flow_shift(
    screen: np.ndarray,
    shift_x_pix: int = 0,
    shift_y_pix: int = 0,
    mask: np.ndarray | None = None,
    remove_mean: bool = True,
) -> np.ndarray:
    """
    Shift a phase screen with periodic boundary conditions.

    This is useful for a toy frozen-flow atmosphere model.
    """
    shifted = np.roll(
        np.roll(np.nan_to_num(screen, nan=0.0), int(shift_y_pix), axis=0),
        int(shift_x_pix),
        axis=1,
    )

    if mask is not None:
        shifted = np.where(mask, shifted, np.nan)
        if remove_mean:
            shifted = remove_piston(shifted, mask)

    return shifted


def frozen_flow_shift_physical(
    screen: np.ndarray,
    vx: float,
    vy: float,
    dt: float,
    delta: float,
    mask: np.ndarray | None = None,
    remove_mean: bool = True,
) -> np.ndarray:
    """
    Shift a phase screen using physical wind velocity.

    Parameters
    ----------
    vx, vy:
        Wind velocity in m/s.
    dt:
        Time in seconds relative to the initial phase screen.
    delta:
        Phase-screen sampling in m/pixel.
    mask:
        Optional pupil mask. If given, the shifted screen is re-masked and
        piston is removed inside the pupil.
    """
    if delta <= 0:
        raise ValueError("delta must be positive.")

    shift_x_pix = int(np.round(vx * dt / delta))
    shift_y_pix = int(np.round(vy * dt / delta))

    return frozen_flow_shift(
        screen,
        shift_x_pix=shift_x_pix,
        shift_y_pix=shift_y_pix,
        mask=mask,
        remove_mean=remove_mean,
    )
