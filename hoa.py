import numpy as np
from einops import rearrange
from scipy.special import sph_harm


def real_sh(azi: np.ndarray, col: np.ndarray, order: int) -> np.ndarray:
    r"""Calculate harmonic bases Y_nm(θ, φ).

    Args:
        order: int
        azi: np.ndarray, azimuth, [0, 2π)
        col: np.ndarray, colatitude [0, pi)

    Outputs:
        bases: (θ.shape, chn), bases Y_nm(θ, φ), where chn = (order+1)^2
    """

    bases = []

    for n in range(order + 1):
        for m in range(-n, n + 1):
            Y = sph_harm(m, n, azi, col)  # θ.shape
            bases.append(Y.real)  # (chn, θ.shape)
    
    bases = np.stack(bases, axis=0)  # (chn, θ.shape), Y_nm(θ, φ)

    return bases


def forward_hoa(value: np.ndarray, azi: np.ndarray, col: np.ndarray, order: int) -> np.ndarray:
    r"""Calculate the HOA coefficients a_nm of signals.

    a_nm = \int_{f(θ, φ) Y_nm(θ, φ) sinθ dθdφ}

    Args:
        value: value at azi and col
        azi: elevation
        col: azimuth
        order: int

    Outputs:
        a_nm: (chn,) where chn = (order+1)^2
    """
    
    # Y_nm(θ, φ)
    bases = real_sh(azi, col, order)  # (chn, θ.shape)

    # a_nm = \int_{f(θ, φ) Y_nm(θ, φ) sinθ dθdφ}
    C = bases.shape[0]
    hoa = (value * bases).reshape(C, -1).sum(axis=1)  # (chn,)

    return hoa


def inverse_hoa(hoa: np.ndarray, azi: np.ndarray, col: np.ndarray, order: int) -> np.ndarray:
    r"""Inverse spherical transform. Input HOA coefficients and output 
    reconstructed signal.

    f(θ, φ) = \sum_{n,m}{a_{nm} Y_nm(θ, φ)}
    
    Args:
        hoa: (chn,)
        azi: np.ndarray
        col: np.ndarray
        order: int

    Outputs:
        out: θ.shape, reconstructed signal
    """

    # Y_nm(θ, φ)
    bases = real_sh(azi, col, order)  # (chn, θ.shape)

    # f(θ, φ) = \sum_{n,m}{a_{nm} Y_nm(θ, φ)}
    bases = rearrange(bases, 'c ... -> ... c')  # (θ.shape, chn)
    out = np.dot(bases, hoa)  # θ.shape

    return out


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