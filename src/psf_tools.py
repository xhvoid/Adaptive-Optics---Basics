import numpy as np


def compute_psf_from_phase(phase_rad, mask, pad_factor=4):
    """
    Compute focal-plane PSF from pupil-plane phase.
    """
    N = phase_rad.shape[0]

    pupil_field = np.zeros_like(phase_rad, dtype=complex)
    phase_clean = np.nan_to_num(phase_rad, nan=0.0)

    pupil_field[mask] = np.exp(1j * phase_clean[mask])

    pad_N = pad_factor * N
    padded = np.zeros((pad_N, pad_N), dtype=complex)

    start = (pad_N - N) // 2
    end = start + N

    padded[start:end, start:end] = pupil_field

    focal_field = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(padded)
        )
    )

    psf = np.abs(focal_field) ** 2
    psf /= np.sum(psf)

    return psf


def strehl_ratio(phase_rad, mask, pad_factor=4):
    """
    Compute approximate Strehl ratio.
    """
    psf_aberrated = compute_psf_from_phase(
        phase_rad,
        mask,
        pad_factor=pad_factor,
    )

    psf_ideal = compute_psf_from_phase(
        np.zeros_like(phase_rad),
        mask,
        pad_factor=pad_factor,
    )

    return np.max(psf_aberrated) / np.max(psf_ideal)


def radial_profile(psf, center=None):
    """
    Compute azimuthally averaged PSF radial profile.
    """
    ny, nx = psf.shape

    if center is None:
        center = (nx // 2, ny // 2)

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    r = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    r_int = r.astype(int)

    profile = np.bincount(
        r_int.ravel(),
        weights=psf.ravel(),
    ) / np.bincount(r_int.ravel())

    radius = np.arange(len(profile))

    return radius, profile