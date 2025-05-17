import numpy as np
from scipy.special import sph_harm
from einops import rearrange, reduce


def forward_hoa(value: np.ndarray, azi: np.ndarray, col: np.ndarray, order: int) -> np.ndarray:
    r"""Calculate the HOA coefficients a_nm of signals.

    a_nm = \int_{f(θ, φ) Y_nm(θ, φ) sinθ dθdφ}

    Args:
        value: (n, ...)
        azi: (n, ...), azimuth, [0, 2π)
        col: (n, ...), colatitude, [0, π]. (col = π/2 - elevation)
        order: int, HOA order

    Outputs:
        a_nm: (chn,) where chn = (order+1)^2
    """
    
    # Y_nm(θ, φ)
    bases = real_sh(azi, col, order)  # (chn, n, ...)

    # a_nm = \int_{f(θ, φ) Y_nm(θ, φ) sinθ dθdφ}
    hoa = reduce(bases * value, 'c ... -> c', reduction='sum')  # (chn,)
    
    return hoa


def inverse_hoa(hoa: np.ndarray, azi: np.ndarray, col: np.ndarray, order: int) -> np.ndarray:
    r"""Inverse spherical transform. Input HOA coefficients and output 
    reconstructed signal.

    f(θ, φ) = \sum_{n,m}{a_{nm} Y_nm(θ, φ)}
    
    Args:
        hoa: (chn,), HOA coefficients
        azi: (n, ...), azimuth, [0, 2π)
        col: (n, ...), colatitude, [0, π]. (col = π/2 - elevation)
        order: int, HOA order

    Outputs:
        out: (n, ...), reconstructed signal
    """

    # Y_nm(θ, φ)
    bases = real_sh(azi, col, order)  # (chn, n, ...)

    # f(θ, φ) = \sum_{n,m}{a_{nm} Y_nm(θ, φ)}
    bases = rearrange(bases, 'c ... -> ... c')  # (n, ..., chn)
    out = np.dot(bases, hoa)  # (n, ...)

    return out


def real_sh(azi: np.ndarray, col: np.ndarray, order: int) -> np.ndarray:
    r"""Calculate harmonic bases Y_nm(θ, φ).

    Args:
        order: int
        azi: (n, ...), azimuth, [0, 2π)
        col: (n, ...), colatitude [0, π]

    Outputs:
        bases: (n, ..., chn), bases Y_nm(θ, φ), where chn = (order+1)^2
    """

    bases = []

    for n in range(order + 1):
        for m in range(-n, n + 1):
            Y = sph_harm(m, n, azi, col)  # (n, ...)
            bases.append(Y.real)  # (chn, n, ...)
    
    # Y_nm(θ, φ)
    bases = np.stack(bases, axis=0)  # (chn, n, ...)

    return bases


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Forward
    order = 5

    azi = np.deg2rad([20, 180])
    col = np.deg2rad([80, 120])
    value = np.array([1.0, 0.8])

    # Calculate HOA coefficient
    hoa = forward_hoa(value=value, azi=azi, col=col, order=order)  # (order+1)^2
    print("HOA channels:", hoa.shape)

    # Inverse HOA transform for visualization
    azi = np.deg2rad(np.arange(0, 360))
    col = np.deg2rad(np.arange(0, 180))
    azi, col = np.meshgrid(azi, col)

    recon = inverse_hoa(hoa, azi, col, order)  # (360, 180)

    # Visualization
    plt.matshow(recon, origin='upper', aspect='auto', cmap='jet')
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Colatitude (deg)")
    plt.savefig("hoa.pdf")
    print("Write out to hoa.pdf")