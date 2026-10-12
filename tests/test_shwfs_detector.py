import numpy as np
import pytest

from shwfs_detector import (
    add_detector_noise,
    build_detector_response_matrix,
    centroid,
    measure_centroid_shifts,
    reconstruct_from_centroid_shifts,
)
from zernike import make_pupil_grid


def test_centroid_of_symmetric_spot_is_near_zero():
    coords = np.arange(9) - 4
    x, y = np.meshgrid(coords, coords)
    spot = np.exp(-(x**2 + y**2) / 3.0)

    cx, cy = centroid(spot)

    assert cx == pytest.approx(0.0, abs=1e-14)
    assert cy == pytest.approx(0.0, abs=1e-14)


def test_zero_image_centroid_returns_nan():
    cx, cy = centroid(np.zeros((5, 5)))

    assert np.isnan(cx)
    assert np.isnan(cy)


def test_negative_detector_noise_inputs_raise_error():
    spot = np.ones((3, 3)) / 9.0

    with pytest.raises(ValueError, match="read_noise_e"):
        add_detector_noise(spot, read_noise_e=-1.0)
    with pytest.raises(ValueError, match="background_e"):
        add_detector_noise(spot, background_e=-1.0)
    with pytest.raises(ValueError, match="photons"):
        add_detector_noise(spot, photons=-1.0)


def test_zero_phase_centroid_shifts_are_near_zero():
    x, y, _, _, mask, _ = make_pupil_grid(N=48, diameter=1.0)
    phase = np.zeros_like(x)

    _, shifts = measure_centroid_shifts(
        phase,
        mask,
        x,
        y,
        n_lenslets=4,
        min_fill=0.4,
        pad_factor=2,
        detector_window_size=16,
        photons=None,
        seed=1,
    )

    assert np.all(np.isfinite(shifts))
    assert np.max(np.abs(shifts)) < 1e-12


def test_one_sided_small_poke_recovers_an_independent_tilt_amplitude():
    x, y, _, _, mask, _ = make_pupil_grid(N=64, diameter=1.0)
    tilt = np.where(mask, 10.0 * x, np.nan)
    settings = dict(n_lenslets=4, pad_factor=4, detector_window_size=8)
    response, _, centers, masks, reference = build_detector_response_matrix(
        {"tilt": tilt},
        mask,
        x,
        y,
        calibration_amplitude=1e-3,
        differential=False,
        **settings,
    )
    actual_amplitude = 4e-4
    _, shifts = measure_centroid_shifts(
        actual_amplitude * tilt,
        mask,
        x,
        y,
        centers=centers,
        masks=masks,
        reference=reference,
        **settings,
    )

    recovered, *_ = reconstruct_from_centroid_shifts(shifts, response)

    assert recovered[0] == pytest.approx(actual_amplitude, rel=2e-4)
