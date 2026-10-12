import numpy as np
import pytest

from phase_screen import (
    fourier_phase_screen,
    frozen_flow_shift,
    opd_to_phase,
    phase_to_opd,
    pupil_grid,
    r0_from_seeing,
    rms,
    scale_r0_with_wavelength,
    von_karman_phase_variance,
)


def test_r0_scales_with_wavelength_to_six_fifths_power():
    r0_ref = r0_from_seeing(0.8, wavelength=500e-9)
    r0_h = scale_r0_with_wavelength(r0_ref, wavelength=1.0e-6, wavelength_ref=500e-9)

    assert r0_h == pytest.approx(r0_ref * 2.0 ** (6.0 / 5.0))


def test_phase_opd_roundtrip_preserves_values():
    phase = np.array([[0.0, 0.5], [-1.2, np.nan]])
    wavelength = 700e-9

    opd = phase_to_opd(phase, wavelength)
    recovered = opd_to_phase(opd, wavelength)

    assert np.allclose(recovered, phase, equal_nan=True)


def test_fourier_phase_screen_reaches_target_rms():
    phase, _, _, mask = fourier_phase_screen(
        N=64,
        delta=0.02,
        r0=0.15,
        diameter=1.0,
        seed=4,
        target_rms_rad=1.25,
        normalize_rms=True,
    )

    assert phase.shape == (64, 64)
    assert rms(phase, mask) == pytest.approx(1.25, rel=1e-6)
    assert np.all(np.isnan(phase[~mask]))


def test_invalid_physical_inputs_raise_error():
    with pytest.raises(ValueError, match="seeing_arcsec"):
        r0_from_seeing(0.0)
    with pytest.raises(ValueError, match="wavelength"):
        phase_to_opd(np.array([1.0]), 0.0)


def test_kolmogorov_variance_matches_noll_coefficients():
    scale = (1.0 / 0.1) ** (5.0 / 3.0)

    piston_removed = von_karman_phase_variance(1.0, 0.1, L0=None)
    tilt_removed = von_karman_phase_variance(1.0, 0.1, L0=None, remove_tilt=True)

    assert piston_removed / scale == pytest.approx(1.0299, rel=0.01)
    assert tilt_removed / scale == pytest.approx(0.134, rel=0.02)


def test_outer_scale_reduces_aperture_variance():
    kolmogorov = von_karman_phase_variance(1.0, 0.1, L0=None)
    von_karman = von_karman_phase_variance(1.0, 0.1, L0=10.0)

    assert von_karman < kolmogorov


def _pupil_variances(phase, x, y, mask):
    values = phase[mask]
    plane = np.column_stack([np.ones(values.size), x[mask], y[mask]])
    tilt_removed = values - plane @ np.linalg.lstsq(plane, values, rcond=None)[0]
    return np.var(values), np.var(tilt_removed)


def test_screen_ensemble_matches_von_karman_theory():
    diameter, r0, L0 = 1.0, 0.1, 25.0
    piston_removed, tilt_removed = [], []
    for seed in range(300):
        phase, x, y, mask = fourier_phase_screen(
            N=64, delta=diameter / 64, r0=r0, L0=L0, diameter=diameter, seed=seed
        )
        total, high_order = _pupil_variances(phase, x, y, mask)
        piston_removed.append(total)
        tilt_removed.append(high_order)

    assert np.mean(piston_removed) == pytest.approx(
        von_karman_phase_variance(diameter, r0, L0), rel=0.10
    )
    assert np.mean(tilt_removed) == pytest.approx(
        von_karman_phase_variance(diameter, r0, L0, remove_tilt=True), rel=0.10
    )


def test_unnormalized_screen_scales_with_r0():
    weak, *_ = fourier_phase_screen(N=64, delta=1.0 / 64, r0=0.2, seed=3)
    strong, *_ = fourier_phase_screen(N=64, delta=1.0 / 64, r0=0.1, seed=3)

    assert np.nanstd(strong) / np.nanstd(weak) == pytest.approx(2.0 ** (5.0 / 6.0))


def test_frozen_flow_window_follows_the_wind():
    sheet = np.random.default_rng(0).normal(size=(80, 80))
    _, _, mask = pupil_grid(32, 1.0, None)

    start = frozen_flow_shift(sheet, 0, 0, mask=mask, remove_mean=False)
    moved = frozen_flow_shift(sheet, 3, -2, mask=mask, remove_mean=False)

    # Content moves +3 px in x and -2 px in y: moved[i, j] = start[i + 2, j - 3].
    assert np.array_equal(moved[:-2, 3:], start[2:, :-3])


def test_frozen_flow_rejects_unsafe_shifts():
    masked, _, _, mask = fourier_phase_screen(N=32, delta=1.0 / 32, r0=0.2, seed=1)
    with pytest.raises(ValueError, match="NaN border"):
        frozen_flow_shift(masked, 1, 0, mask=mask)

    sheet = np.zeros((40, 40))
    _, _, small_mask = pupil_grid(32, 1.0, None)
    with pytest.raises(ValueError, match="left the phase sheet"):
        frozen_flow_shift(sheet, 5, 0, mask=small_mask)
