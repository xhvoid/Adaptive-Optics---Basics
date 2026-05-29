"""
Detector-level Shack-Hartmann wavefront sensor utilities.

This module extends the geometric SH-WFS model in ``reconstruction.py`` by
simulating diffraction-limited lenslet spots, detector noise, centroiding, and
centroid-shift-based response matrices.

The calibration strategy used here is practical for a learning simulator:
centroid shifts are kept in detector-pixel units, and the response matrix maps
modal coefficients or actuator commands directly to centroid shifts. Therefore
one does not need to explicitly convert centroid shift to physical wavefront
slope in radians or meters; the response matrix provides the calibration.
"""

from __future__ import annotations

import numpy as np

try:
    from reconstruction import subaperture_masks
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "shwfs_detector.py expects reconstruction.py to be available "
        "in the same project directory."
    ) from exc


def _bounding_box(mask: np.ndarray, margin: int = 0) -> tuple[slice, slice]:
    """Return a rectangular bounding box around a boolean mask."""
    yy, xx = np.where(mask)
    if yy.size == 0:
        raise ValueError("Cannot build a bounding box for an empty mask.")

    y0 = max(int(yy.min()) - margin, 0)
    y1 = min(int(yy.max()) + margin + 1, mask.shape[0])
    x0 = max(int(xx.min()) - margin, 0)
    x1 = min(int(xx.max()) + margin + 1, mask.shape[1])

    return slice(y0, y1), slice(x0, x1)


def lenslet_spot_from_phase(
    phase_rad: np.ndarray,
    lenslet_mask: np.ndarray,
    pad_factor: int = 8,
    remove_local_piston: bool = True,
) -> np.ndarray:
    """
    Compute a diffraction spot from one Shack-Hartmann lenslet.

    Parameters
    ----------
    phase_rad:
        Pupil-plane phase in radians.
    lenslet_mask:
        Boolean mask selecting one sub-aperture/lenslet footprint.
    pad_factor:
        Zero-padding factor for the local Fourier transform. Larger values give
        smoother centroid shifts but cost more CPU time.
    remove_local_piston:
        Remove the mean phase inside the sub-aperture before computing the
        spot. Local piston does not affect the intensity spot.

    Returns
    -------
    spot:
        Normalized detector-plane intensity image for the lenslet.
    """
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1.")

    ys, xs = _bounding_box(lenslet_mask)
    local_mask = lenslet_mask[ys, xs]
    local_phase = np.nan_to_num(phase_rad[ys, xs], nan=0.0).astype(float)

    if remove_local_piston and np.any(local_mask):
        local_phase = local_phase.copy()
        local_phase[local_mask] -= np.mean(local_phase[local_mask])

    local_field = np.zeros(local_phase.shape, dtype=complex)
    local_field[local_mask] = np.exp(1j * local_phase[local_mask])

    ny, nx = local_field.shape
    pad_ny = int(pad_factor * ny)
    pad_nx = int(pad_factor * nx)

    padded = np.zeros((pad_ny, pad_nx), dtype=complex)
    y0 = (pad_ny - ny) // 2
    x0 = (pad_nx - nx) // 2
    padded[y0 : y0 + ny, x0 : x0 + nx] = local_field

    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    spot = np.abs(focal_field) ** 2
    total = spot.sum()
    if total > 0:
        spot = spot / total

    return spot


def add_detector_noise(
    normalized_spot: np.ndarray,
    photons: float | None = None,
    read_noise_e: float = 0.0,
    background_e: float = 0.0,
    seed: int | None = None,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Add photon noise, read noise, and background to a normalized spot.

    Parameters
    ----------
    normalized_spot:
        Spot image normalized to unit total flux.
    photons:
        Expected number of detected photons/electrons in the spot. If None, no
        Poisson noise is applied and the input spot is simply returned as a
        floating point image, plus optional Gaussian read noise/background.
    read_noise_e:
        Gaussian read noise standard deviation in electrons per pixel.
    background_e:
        Constant background level in electrons per pixel.
    seed:
        Random seed.
    clip_negative:
        Clip negative noisy pixel values to zero.

    Returns
    -------
    image:
        Noisy detector image in electron-count units if photons is not None;
        otherwise an arbitrary floating point image.
    """
    rng = np.random.default_rng(seed)
    spot = np.asarray(normalized_spot, dtype=float)

    if photons is None:
        image = spot.copy()
        if background_e != 0:
            image = image + background_e
        if read_noise_e > 0:
            image = image + rng.normal(scale=read_noise_e, size=spot.shape)
    else:
        if photons < 0:
            raise ValueError("photons must be non-negative or None.")
        expected = photons * spot + background_e
        image = rng.poisson(expected).astype(float)
        if read_noise_e > 0:
            image = image + rng.normal(scale=read_noise_e, size=spot.shape)

    if clip_negative:
        image = np.maximum(image, 0.0)

    return image


def centroid(
    image: np.ndarray,
    threshold_fraction: float = 0.0,
    subtract_minimum: bool = False,
) -> tuple[float, float]:
    """
    Compute a center-of-gravity centroid in detector-pixel coordinates.

    The returned coordinate origin is the geometric center of the image. A
    positive x centroid is to the right, and a positive y centroid is upward in
    mathematical coordinates.
    """
    img = np.asarray(image, dtype=float).copy()

    if subtract_minimum:
        img -= np.nanmin(img)
        img = np.maximum(img, 0.0)

    if threshold_fraction > 0:
        threshold = threshold_fraction * np.nanmax(img)
        img[img < threshold] = 0.0

    total = np.nansum(img)
    if total <= 0 or not np.isfinite(total):
        return np.nan, np.nan

    ny, nx = img.shape
    x = np.arange(nx) - (nx - 1) / 2
    y = (ny - 1) / 2 - np.arange(ny)  # upward-positive y
    X, Y = np.meshgrid(x, y)

    cx = float(np.nansum(img * X) / total)
    cy = float(np.nansum(img * Y) / total)

    return cx, cy


def reference_centroids(
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    pad_factor: int = 8,
    threshold_fraction: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Compute zero-phase reference centroids for all valid lenslets.
    """
    centers, masks = subaperture_masks(
        X,
        Y,
        pupil_mask,
        n_lenslets=n_lenslets,
        min_fill=min_fill,
    )

    zero_phase = np.zeros_like(X, dtype=float)
    refs = []
    for m in masks:
        spot0 = lenslet_spot_from_phase(zero_phase, m, pad_factor=pad_factor)
        refs.append(centroid(spot0, threshold_fraction=threshold_fraction))

    return centers, masks, np.asarray(refs, dtype=float)


def measure_centroid_shifts(
    phase_rad: np.ndarray,
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    pad_factor: int = 8,
    photons: float | None = None,
    read_noise_e: float = 0.0,
    background_e: float = 0.0,
    threshold_fraction: float = 0.0,
    seed: int | None = 1,
    reference: np.ndarray | None = None,
    masks: list[np.ndarray] | None = None,
    centers: np.ndarray | None = None,
    return_spots: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Measure centroid shifts for a detector-level SH-WFS.

    Returns shifts in detector-pixel units relative to the zero-phase reference
    centroid of each lenslet. These shifts can be used directly as a calibrated
    measurement vector when the response matrix is built with the same function.
    """
    if masks is None or centers is None or reference is None:
        centers, masks, reference = reference_centroids(
            pupil_mask,
            X,
            Y,
            n_lenslets=n_lenslets,
            min_fill=min_fill,
            pad_factor=pad_factor,
            threshold_fraction=threshold_fraction,
        )

    rng = np.random.default_rng(seed)
    shifts = []
    spots = []

    for i, m in enumerate(masks):
        spot = lenslet_spot_from_phase(
            phase_rad,
            m,
            pad_factor=pad_factor,
            remove_local_piston=True,
        )
        noisy = add_detector_noise(
            spot,
            photons=photons,
            read_noise_e=read_noise_e,
            background_e=background_e,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        c = np.asarray(centroid(noisy, threshold_fraction=threshold_fraction))
        shifts.append(c - reference[i])
        if return_spots:
            spots.append(noisy)

    shifts = np.asarray(shifts, dtype=float)

    if return_spots:
        return centers, shifts, spots
    return centers, shifts


def build_detector_response_matrix(
    modes: dict[str, np.ndarray],
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    pad_factor: int = 8,
    threshold_fraction: float = 0.0,
) -> tuple[np.ndarray, list[str], np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Build a centroid-shift response matrix for detector-level SH-WFS.

    Each column is the flattened centroid-shift vector generated by one unit
    coefficient of one input mode.
    """
    names = list(modes.keys())
    centers, masks, reference = reference_centroids(
        pupil_mask,
        X,
        Y,
        n_lenslets=n_lenslets,
        min_fill=min_fill,
        pad_factor=pad_factor,
        threshold_fraction=threshold_fraction,
    )

    columns = []
    for name in names:
        _, shifts = measure_centroid_shifts(
            modes[name],
            pupil_mask,
            X,
            Y,
            n_lenslets=n_lenslets,
            min_fill=min_fill,
            pad_factor=pad_factor,
            photons=None,
            read_noise_e=0.0,
            background_e=0.0,
            threshold_fraction=threshold_fraction,
            seed=1,
            reference=reference,
            masks=masks,
            centers=centers,
        )
        columns.append(shifts.reshape(-1))

    A = np.column_stack(columns)
    return A, names, centers, masks, reference


def reconstruct_from_centroid_shifts(
    shifts: np.ndarray,
    response_matrix: np.ndarray,
    rcond: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Least-squares reconstruction from centroid shifts.
    """
    s = np.asarray(shifts, dtype=float).reshape(-1)
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(
        response_matrix,
        s,
        rcond=rcond,
    )
    return coeffs, residuals, rank, singular_values


def centroid_noise_scan(
    phase_rad: np.ndarray,
    response_matrix: np.ndarray,
    modes: dict[str, np.ndarray],
    names: list[str],
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    photon_levels: list[float],
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    pad_factor: int = 8,
    read_noise_e: float = 0.0,
    background_e: float = 0.0,
    threshold_fraction: float = 0.0,
    n_trials: int = 10,
    seed: int = 1,
    rcond: float = 1e-4,
) -> dict[float, dict[str, float]]:
    """
    Convenience function for scanning photon-noise sensitivity.

    Returns a dictionary with mean/std residual RMS for each photon level.
    """
    from reconstruction import synthesize_from_coefficients, residual_wavefront, rms

    centers, masks, reference = reference_centroids(
        pupil_mask,
        X,
        Y,
        n_lenslets=n_lenslets,
        min_fill=min_fill,
        pad_factor=pad_factor,
        threshold_fraction=threshold_fraction,
    )

    rng = np.random.default_rng(seed)
    out = {}

    for photons in photon_levels:
        residuals = []
        for _ in range(n_trials):
            _, shifts = measure_centroid_shifts(
                phase_rad,
                pupil_mask,
                X,
                Y,
                n_lenslets=n_lenslets,
                min_fill=min_fill,
                pad_factor=pad_factor,
                photons=photons,
                read_noise_e=read_noise_e,
                background_e=background_e,
                threshold_fraction=threshold_fraction,
                seed=int(rng.integers(0, 2**31 - 1)),
                reference=reference,
                masks=masks,
                centers=centers,
            )
            coeffs, *_ = reconstruct_from_centroid_shifts(
                shifts,
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
            res = residual_wavefront(phase_rad, W_rec, pupil_mask)
            residuals.append(rms(res, pupil_mask))

        residuals = np.asarray(residuals)
        out[photons] = {
            "mean_residual_rms": float(np.mean(residuals)),
            "std_residual_rms": float(np.std(residuals)),
        }

    return out
