import numpy as np
import pytest

from pwfs_forward import (
    check_pwfs_geometry,
    extract_cutout,
    make_aligned_pupil_mask,
    make_pwfs_grid,
    pwfs_intensity,
    pwfs_signal_maps_from_intensity,
)


def test_masked_phase_matches_zero_filled_phase_and_conserves_flux():
    x, y, _, _, pupil = make_pwfs_grid(n_fft=128, n_pupil=32)
    phase = 0.3 * x + 0.2 * (x**2 + y**2)
    settings = dict(separation=32, modulation_radius=0.5, n_modulation_points=8)

    masked = pwfs_intensity(np.where(pupil, phase, np.nan), pupil, x, y, **settings)
    zero_filled = pwfs_intensity(np.where(pupil, phase, 0.0), pupil, x, y, **settings)

    assert np.all(np.isfinite(masked))
    assert np.allclose(masked, zero_filled)
    assert masked.sum() == pytest.approx(pupil.sum(), rel=1e-12)


@pytest.mark.parametrize("invalid_phase", [np.nan, np.inf, -np.inf])
def test_nonfinite_phase_inside_pupil_is_rejected(invalid_phase):
    x, y, _, _, pupil = make_pwfs_grid(n_fft=128, n_pupil=32)
    phase = np.zeros_like(x)
    phase[64, 64] = invalid_phase

    with pytest.raises(ValueError, match="finite inside the pupil"):
        pwfs_intensity(phase, pupil, x, y, separation=32)


@pytest.mark.parametrize("n_pupil", [32, 33])
def test_pupil_extraction_and_mask_share_the_fft_center(n_pupil):
    x, y, _, _, pupil = make_pwfs_grid(n_fft=128, n_pupil=n_pupil)
    intensity = pwfs_intensity(np.zeros_like(x), pupil, x, y, separation=32)

    sx, sy, total, valid = pwfs_signal_maps_from_intensity(
        intensity, n_pupil=n_pupil, separation=32
    )
    cropped_pupil = extract_cutout(pupil, (64, 64), size=n_pupil)

    assert check_pwfs_geometry(128, n_pupil, 32)
    assert sx.shape == sy.shape == total.shape == valid.shape == (n_pupil, n_pupil)
    assert np.array_equal(valid, cropped_pupil)
    assert np.array_equal(valid, make_aligned_pupil_mask(n_pupil))
    assert np.all(np.isfinite(sx[valid]))
    assert np.all(np.isfinite(sy[valid]))


def test_geometry_check_accounts_for_the_last_pixel_of_an_odd_crop():
    assert check_pwfs_geometry(64, 32, 16)

    with pytest.raises(ValueError, match="clipped"):
        check_pwfs_geometry(64, 33, 16)
    with pytest.raises(ValueError, match="outside the detector"):
        extract_cutout(np.zeros((64, 64)), (48, 48), size=33)
