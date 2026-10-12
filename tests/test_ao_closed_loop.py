import numpy as np
import pytest

from ao_closed_loop import (
    actuator_centers_on_pupil,
    build_dm_wfs_response_matrix,
    fit_dm_to_phase,
    gaussian_influence_functions,
    run_closed_loop_ao,
    synthesize_dm_phase,
    tsvd_reconstructor,
)
from phase_screen import fourier_phase_screen, pupil_grid
from reconstruction import measure_slopes, subaperture_masks


def _small_system():
    x, y, mask = pupil_grid(48, 1.0 / 48, 1.0)
    centers, pitch = actuator_centers_on_pupil(1.0, 7)
    influence = gaussian_influence_functions(x, y, mask, centers, pitch, coupling=0.6)
    return x, y, mask, influence


def test_guard_ring_adds_actuators_just_outside_the_pupil():
    inside, pitch = actuator_centers_on_pupil(1.0, 9)
    guarded, guarded_pitch = actuator_centers_on_pupil(1.0, 9, radius_factor=1.2)

    radii = np.hypot(guarded[:, 0], guarded[:, 1])
    assert guarded_pitch == pytest.approx(pitch)
    assert len(guarded) > len(inside)
    assert radii.max() <= 0.6 + 1e-12
    assert np.any(radii > 0.5)


def test_dm_fit_recovers_a_shape_the_mirror_can_make():
    _, _, mask, influence = _small_system()
    commands = np.random.default_rng(3).normal(size=influence.shape[0])
    phase = synthesize_dm_phase(commands, influence, mask)

    _, fitted, residual = fit_dm_to_phase(phase, influence, mask)

    assert np.nanstd(residual[mask]) < 1e-8 * np.nanstd(phase[mask])
    assert np.allclose(fitted[mask], phase[mask])


def test_zero_response_has_no_controllable_commands():
    response = np.zeros((4, 3))

    reconstructor, singular_values, rank = tsvd_reconstructor(response)

    assert rank == 0
    assert np.all(singular_values == 0.0)
    assert np.array_equal(reconstructor @ np.ones(4), np.zeros(3))


def test_zero_cutoff_keeps_only_numerically_resolvable_modes():
    response = np.diag([1.0, 1e-18, 0.0])

    reconstructor, _, rank = tsvd_reconstructor(response, rcond=0.0)

    assert rank == 1
    assert np.all(np.isfinite(reconstructor))
    assert np.allclose(reconstructor @ np.ones(3), [1.0, 0.0, 0.0])


def test_cached_lenslet_geometry_gives_the_same_slopes():
    x, y, mask, influence = _small_system()
    phase = np.where(mask, np.nan_to_num(influence[5]) + 0.3 * x, np.nan)
    centers, masks = subaperture_masks(x, y, mask, n_lenslets=6, min_fill=0.5)

    _, fresh = measure_slopes(phase, mask, x, y, n_lenslets=6, min_fill=0.5)
    _, cached = measure_slopes(phase, mask, x, y, centers=centers, masks=masks)

    assert np.array_equal(fresh, cached)


@pytest.mark.parametrize("gain, converges", [(0.4, True), (0.9, False)])
def test_two_frame_delay_sets_the_integrator_gain_limit(gain, converges):
    # With d frames of delay a pure integrator is stable only for
    # gain < 2 sin(pi / (2 (2 d + 1))), which is 0.618 for d = 2.
    x, y, mask, influence = _small_system()
    response, _ = build_dm_wfs_response_matrix(influence, mask, x, y, n_lenslets=6, min_fill=0.5)
    screen, *_ = fourier_phase_screen(N=48, delta=1.0 / 48, r0=0.2, seed=2)

    history = run_closed_loop_ao(
        screen,
        mask,
        x,
        y,
        influence,
        response,
        n_steps=60,
        wind_shift_per_step=(0, 0),
        gain=gain,
        n_lenslets=6,
        min_fill=0.5,
        compute_strehl=False,
        delay_frames=2,
    )

    start, end = history["rms_before"][0], history["rms_before"][-1]
    assert (end < 0.6 * start) == converges
    assert (end > 2.0 * start) == (not converges)


@pytest.mark.parametrize("gain", [0.9, 1.0, 1.1])
def test_one_frame_delay_matches_the_decay_and_oscillation_boundary(gain):
    x, y, mask, influence = _small_system()
    response, _ = build_dm_wfs_response_matrix(influence, mask, x, y, n_lenslets=6)
    reconstructor, _, _ = tsvd_reconstructor(response)
    # The strongest right singular vector is an observable, exactly DM-fit phase.
    commands = np.linalg.svd(response, full_matrices=False)[2][0]
    phase = synthesize_dm_phase(commands, influence, mask)
    history = run_closed_loop_ao(
        phase, mask, x, y, influence, response,
        n_steps=60, wind_shift_per_step=(0, 0), gain=gain,
        n_lenslets=6, delay_frames=1, reconstructor=reconstructor,
        compute_strehl=False, record_every=1,
    )
    samples = phase[mask]
    measured = history["residual_maps"][:, mask] @ samples / (samples @ samples)
    # e[k+1] = e[k] - g*e[k-1]; poles of z**2-z+g have modulus sqrt(g).
    expected = np.ones(60)
    for k in range(1, 59):
        expected[k + 1] = expected[k] - gain * expected[k - 1]

    assert np.allclose(measured, expected, rtol=1e-8, atol=1e-10)
    if gain == 1.0:
        # At the boundary the signed error repeats every six frames, without decay.
        assert np.allclose(measured[-6:], measured[:6], atol=1e-10)
    else:
        grows = np.linalg.norm(measured[-6:]) > np.linalg.norm(measured[:6])
        assert grows == (gain > 1.0)


def test_precomputed_reconstructor_matches_per_frame_least_squares():
    x, y, mask, influence = _small_system()
    response, _ = build_dm_wfs_response_matrix(influence, mask, x, y, n_lenslets=6, min_fill=0.5)
    sheet, *_ = fourier_phase_screen(N=96, delta=1.0 / 48, r0=0.2, diameter=None, seed=4)
    common = dict(
        n_steps=20,
        wind_shift_per_step=(1, 0),
        gain=0.5,
        n_lenslets=6,
        min_fill=0.5,
        rcond=1e-3,
        compute_strehl=False,
        delay_frames=1,
    )

    per_frame = run_closed_loop_ao(sheet, mask, x, y, influence, response, **common)
    R, _, _ = tsvd_reconstructor(response, rcond=1e-3)
    precomputed = run_closed_loop_ao(sheet, mask, x, y, influence, response, reconstructor=R, **common)

    assert np.allclose(per_frame["rms_before"], precomputed["rms_before"], rtol=1e-8)


def test_recorded_residual_maps_match_the_rms_history():
    x, y, mask, influence = _small_system()
    response, _ = build_dm_wfs_response_matrix(influence, mask, x, y, n_lenslets=6, min_fill=0.5)
    sheet, *_ = fourier_phase_screen(N=96, delta=1.0 / 48, r0=0.2, diameter=None, seed=4)

    history = run_closed_loop_ao(
        sheet,
        mask,
        x,
        y,
        influence,
        response,
        n_steps=12,
        n_lenslets=6,
        min_fill=0.5,
        compute_strehl=False,
        record_every=5,
    )

    assert list(history["residual_frames"]) == [0, 5, 10]
    for frame, residual in zip(history["residual_frames"], history["residual_maps"]):
        assert np.nanstd(residual[mask]) == pytest.approx(history["rms_before"][frame])
