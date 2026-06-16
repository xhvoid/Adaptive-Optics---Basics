"""
Simple closed-loop adaptive-optics utilities.

This module provides both an ideal geometric SH-WFS closed-loop baseline and a
more realistic detector-level centroiding closed-loop extension. It is compact
and readable by design: the goal is to demonstrate AO control concepts, not to
replace a calibrated AO simulation package.
"""

from __future__ import annotations

import numpy as np

from reconstruction import measure_slopes, rms
from psf_tools import strehl_ratio
from phase_screen import frozen_flow_shift_physical
from shwfs_detector import reference_centroids, measure_centroid_shifts


def actuator_centers_on_pupil(
    diameter: float = 1.0,
    n_actuators: int = 8,
    include_edge: bool = True,
) -> tuple[np.ndarray, float]:
    """Generate square-grid DM actuator centers inside a circular pupil."""
    if n_actuators < 2:
        raise ValueError("n_actuators must be >= 2.")
    if diameter <= 0:
        raise ValueError("diameter must be positive.")

    radius = diameter / 2.0
    if include_edge:
        coords = np.linspace(-radius, radius, n_actuators)
    else:
        pitch_tmp = diameter / n_actuators
        coords = np.linspace(-radius + pitch_tmp / 2.0, radius - pitch_tmp / 2.0, n_actuators)

    pitch = float(coords[1] - coords[0]) if len(coords) > 1 else diameter

    centers = []
    for x in coords:
        for y in coords:
            if x**2 + y**2 <= radius**2:
                centers.append([x, y])

    return np.asarray(centers, dtype=float), pitch


def gaussian_influence_functions(
    X: np.ndarray,
    Y: np.ndarray,
    pupil_mask: np.ndarray,
    centers: np.ndarray,
    pitch: float,
    coupling: float = 0.35,
    normalize_peak: bool = True,
) -> np.ndarray:
    """Build Gaussian DM influence functions."""
    if pitch <= 0:
        raise ValueError("pitch must be positive.")
    if coupling <= 0:
        raise ValueError("coupling must be positive.")

    sigma = coupling * pitch
    infl = []
    for x0, y0 in centers:
        g = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma**2))
        g = np.where(pupil_mask, g, np.nan)
        if normalize_peak:
            peak = np.nanmax(g)
            if peak > 0:
                g = g / peak
        infl.append(g)

    return np.asarray(infl, dtype=float)


def synthesize_dm_phase(
    commands: np.ndarray,
    influence_functions: np.ndarray,
    pupil_mask: np.ndarray,
    remove_mean: bool = True,
) -> np.ndarray:
    """Synthesize a DM phase surface from actuator commands."""
    commands = np.asarray(commands, dtype=float)
    if commands.shape[0] != influence_functions.shape[0]:
        raise ValueError("Number of commands must match number of influence functions.")

    dm = np.nansum(commands[:, None, None] * np.nan_to_num(influence_functions), axis=0)
    dm = np.where(pupil_mask, dm, np.nan)

    if remove_mean:
        dm = dm.copy()
        finite = pupil_mask & np.isfinite(dm)
        if np.any(finite):
            dm[finite] -= np.nanmean(dm[finite])

    return dm


def build_dm_wfs_response_matrix(
    influence_functions: np.ndarray,
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a geometric SH-WFS response matrix for DM actuator commands."""
    columns = []
    reference_centers = None

    for infl in influence_functions:
        centers, slopes = measure_slopes(
            infl,
            pupil_mask,
            X,
            Y,
            n_lenslets=n_lenslets,
            min_fill=min_fill,
            noise_std=0.0,
        )
        if reference_centers is None:
            reference_centers = centers
        elif centers.shape != reference_centers.shape or not np.allclose(centers, reference_centers):
            raise RuntimeError("Sub-aperture geometry changed between influence functions.")
        columns.append(slopes.reshape(-1))

    return np.column_stack(columns), reference_centers


def build_dm_detector_response_matrix(
    influence_functions: np.ndarray,
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    pad_factor: int = 8,
    threshold_fraction: float = 0.0,
    subtract_minimum: bool = False,
    detector_window_size: int | None = None,
    calibration_amplitude: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Build detector-level DM-WFS response matrix.

    Each column is the central finite-difference centroid response to one DM
    actuator influence function.
    """
    if calibration_amplitude <= 0:
        raise ValueError("calibration_amplitude must be positive.")

    centers, masks, reference = reference_centroids(
        pupil_mask,
        X,
        Y,
        n_lenslets=n_lenslets,
        min_fill=min_fill,
        pad_factor=pad_factor,
        threshold_fraction=threshold_fraction,
        detector_window_size=detector_window_size,
        subtract_minimum=subtract_minimum,
    )

    columns = []
    for infl in influence_functions:
        infl_clean = np.nan_to_num(infl, nan=0.0)
        phase_p = calibration_amplitude * infl_clean
        phase_m = -calibration_amplitude * infl_clean

        _, shifts_p = measure_centroid_shifts(
            phase_p,
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
            subtract_minimum=subtract_minimum,
            detector_window_size=detector_window_size,
            reference=reference,
            masks=masks,
            centers=centers,
        )
        _, shifts_m = measure_centroid_shifts(
            phase_m,
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
            subtract_minimum=subtract_minimum,
            detector_window_size=detector_window_size,
            reference=reference,
            masks=masks,
            centers=centers,
        )

        column = ((shifts_p - shifts_m) / (2.0 * calibration_amplitude)).reshape(-1)
        columns.append(column)

    B = np.column_stack(columns)
    return B, centers, masks, reference


def reconstruct_dm_delta(
    slopes: np.ndarray,
    dm_response_matrix: np.ndarray,
    rcond: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Estimate a DM command increment from residual WFS measurements.

    Non-finite measurement rows are dropped together with corresponding response
    matrix rows. This is important for detector-level centroid failures.
    """
    s = np.asarray(slopes, dtype=float).reshape(-1)
    A = np.asarray(dm_response_matrix, dtype=float)

    if A.shape[0] != s.size:
        raise ValueError("Measurement vector length must match response-matrix row count.")

    valid = np.isfinite(s) & np.all(np.isfinite(A), axis=1)
    if not np.any(valid):
        raise ValueError("No finite WFS measurements are available for DM reconstruction.")

    delta, residuals, rank, singular_values = np.linalg.lstsq(A[valid], s[valid], rcond=rcond)
    return delta, residuals, rank, singular_values


def shifted_atmosphere(
    phase0: np.ndarray,
    pupil_mask: np.ndarray,
    shift_x_pix: int = 0,
    shift_y_pix: int = 0,
) -> np.ndarray:
    """Toy frozen-flow phase shift with periodic boundary conditions."""
    full = np.nan_to_num(phase0, nan=0.0)
    shifted = np.roll(np.roll(full, int(shift_y_pix), axis=0), int(shift_x_pix), axis=1)
    shifted = np.where(pupil_mask, shifted, np.nan)
    shifted[pupil_mask] -= np.nanmean(shifted[pupil_mask])
    return shifted


def run_closed_loop_ao(
    phase0: np.ndarray,
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    influence_functions: np.ndarray,
    dm_response_matrix: np.ndarray,
    n_steps: int = 40,
    wind_shift_per_step: tuple[int, int] = (1, 0),
    gain: float = 0.4,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    slope_noise_std: float = 0.0,
    rcond: float = 1e-3,
    command_leak: float = 0.0,
    compute_strehl: bool = True,
    pad_factor: int = 4,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Run a simple geometric-SH-WFS integrator closed-loop AO simulation."""
    if not (0 <= command_leak < 1):
        raise ValueError("command_leak must satisfy 0 <= command_leak < 1.")

    rng = np.random.default_rng(seed)
    commands = np.zeros(influence_functions.shape[0], dtype=float)
    sx_step, sy_step = wind_shift_per_step

    hist = {
        "rms_atmosphere": [],
        "rms_before": [],
        "rms_after": [],
        "strehl_before": [],
        "strehl_after": [],
        "command_norm": [],
    }

    for k in range(n_steps):
        atm = shifted_atmosphere(
            phase0,
            pupil_mask,
            shift_x_pix=int(k * sx_step),
            shift_y_pix=int(k * sy_step),
        )

        dm_before = synthesize_dm_phase(commands, influence_functions, pupil_mask)
        residual_before = np.where(pupil_mask, atm - dm_before, np.nan)
        residual_before[pupil_mask] -= np.nanmean(residual_before[pupil_mask])

        _, slopes = measure_slopes(
            residual_before,
            pupil_mask,
            X,
            Y,
            n_lenslets=n_lenslets,
            min_fill=min_fill,
            noise_std=slope_noise_std,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        delta, *_ = reconstruct_dm_delta(slopes, dm_response_matrix, rcond=rcond)
        commands = (1.0 - command_leak) * commands + gain * delta

        dm_after = synthesize_dm_phase(commands, influence_functions, pupil_mask)
        residual_after = np.where(pupil_mask, atm - dm_after, np.nan)
        residual_after[pupil_mask] -= np.nanmean(residual_after[pupil_mask])

        hist["rms_atmosphere"].append(rms(atm, pupil_mask))
        hist["rms_before"].append(rms(residual_before, pupil_mask))
        hist["rms_after"].append(rms(residual_after, pupil_mask))
        hist["command_norm"].append(float(np.linalg.norm(commands)))

        if compute_strehl:
            hist["strehl_before"].append(float(strehl_ratio(residual_before, pupil_mask, pad_factor=pad_factor)))
            hist["strehl_after"].append(float(strehl_ratio(residual_after, pupil_mask, pad_factor=pad_factor)))

    return {key: np.asarray(value) for key, value in hist.items()}


def run_closed_loop_ao_detector(
    phase0: np.ndarray,
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    influence_functions: np.ndarray,
    dm_detector_response_matrix: np.ndarray,
    detector_reference: np.ndarray,
    detector_masks: list[np.ndarray],
    detector_centers: np.ndarray,
    n_steps: int = 40,
    vx: float = 10.0,
    vy: float = 0.0,
    dt: float = 0.002,
    delta: float = 0.02,
    gain: float = 0.3,
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    pad_factor: int = 8,
    photons: float | None = 1e4,
    read_noise_e: float = 1.0,
    background_e: float = 0.0,
    threshold_fraction: float = 0.0,
    subtract_minimum: bool = False,
    detector_window_size: int | None = None,
    rcond: float = 1e-3,
    command_leak: float = 0.0,
    seed: int = 1,
    compute_strehl: bool = True,
) -> dict[str, np.ndarray]:
    """
    Detector-level closed-loop AO.

    Chain:
        atmosphere -> residual phase -> lenslet spots -> noisy centroids
        -> centroid-response DM reconstruction -> integrator update
    """
    if not (0 <= command_leak < 1):
        raise ValueError("command_leak must satisfy 0 <= command_leak < 1.")

    rng = np.random.default_rng(seed)
    commands = np.zeros(influence_functions.shape[0], dtype=float)

    hist = {
        "rms_atmosphere": [],
        "rms_before": [],
        "rms_after": [],
        "strehl_before": [],
        "strehl_after": [],
        "command_norm": [],
        "valid_centroids": [],
    }

    for k in range(n_steps):
        atm = frozen_flow_shift_physical(
            phase0,
            vx=vx,
            vy=vy,
            dt=k * dt,
            delta=delta,
            mask=pupil_mask,
        )

        dm_before = synthesize_dm_phase(commands, influence_functions, pupil_mask)
        residual_before = np.where(pupil_mask, atm - dm_before, np.nan)
        residual_before[pupil_mask] -= np.nanmean(residual_before[pupil_mask])

        _, shifts, diag = measure_centroid_shifts(
            residual_before,
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
            subtract_minimum=subtract_minimum,
            detector_window_size=detector_window_size,
            seed=int(rng.integers(0, 2**31 - 1)),
            reference=detector_reference,
            masks=detector_masks,
            centers=detector_centers,
            return_diagnostics=True,
        )

        delta_cmd, *_ = reconstruct_dm_delta(shifts, dm_detector_response_matrix, rcond=rcond)
        commands = (1.0 - command_leak) * commands + gain * delta_cmd

        dm_after = synthesize_dm_phase(commands, influence_functions, pupil_mask)
        residual_after = np.where(pupil_mask, atm - dm_after, np.nan)
        residual_after[pupil_mask] -= np.nanmean(residual_after[pupil_mask])

        hist["rms_atmosphere"].append(rms(atm, pupil_mask))
        hist["rms_before"].append(rms(residual_before, pupil_mask))
        hist["rms_after"].append(rms(residual_after, pupil_mask))
        hist["command_norm"].append(float(np.linalg.norm(commands)))
        hist["valid_centroids"].append(int(diag["n_valid"]))

        if compute_strehl:
            hist["strehl_before"].append(float(strehl_ratio(residual_before, pupil_mask, pad_factor=pad_factor)))
            hist["strehl_after"].append(float(strehl_ratio(residual_after, pupil_mask, pad_factor=pad_factor)))

    return {key: np.asarray(value) for key, value in hist.items()}


def gain_scan(
    gains: list[float],
    phase0: np.ndarray,
    pupil_mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    influence_functions: np.ndarray,
    dm_response_matrix: np.ndarray,
    n_steps: int = 40,
    wind_shift_per_step: tuple[int, int] = (1, 0),
    n_lenslets: int = 12,
    min_fill: float = 0.5,
    slope_noise_std: float = 0.0,
    rcond: float = 1e-3,
    command_leak: float = 0.0,
    seed: int = 1,
) -> dict[float, dict[str, float]]:
    """Run a compact loop-gain scan and summarize residual performance."""
    out = {}
    for gain in gains:
        hist = run_closed_loop_ao(
            phase0,
            pupil_mask,
            X,
            Y,
            influence_functions,
            dm_response_matrix,
            n_steps=n_steps,
            wind_shift_per_step=wind_shift_per_step,
            gain=gain,
            n_lenslets=n_lenslets,
            min_fill=min_fill,
            slope_noise_std=slope_noise_std,
            rcond=rcond,
            command_leak=command_leak,
            compute_strehl=False,
            seed=seed,
        )
        tail = hist["rms_after"][max(0, n_steps // 2) :]
        out[float(gain)] = {
            "median_tail_rms": float(np.median(tail)),
            "mean_tail_rms": float(np.mean(tail)),
            "final_rms": float(hist["rms_after"][-1]),
        }
    return out
