import numpy as np


def irm(Ts, Is):
    """
    Generate Ideal Ratio Mask

    Args:
        Ts (np.ndarray):
            Frequency domain representation of the target (nb_of_channels, nb_of_frames, nb_of_bins).
        Is (np.ndarray):
            Frequency domain representation of the interference (nb_of_channels, nb_of_frames, nb_of_bins).

    Returns:
        Ms (np.ndarray):
            Ideal ratio mask (nb_of_channels, nb_of_frames, nb_of_bins).
    """

    return np.clip((np.abs(Ts) ** 2) / (np.abs(Ts) ** 2 + np.abs(Is) ** 2 + 1e-20), a_min=0.0, a_max=1.0)


def ibm(Ts, Is, threshold=0.0):
    """
    Generate Ideal Binary Mask

    Args:
        Ts (np.ndarray):
            Frequency domain representation of the target (nb_of_channels, nb_of_frames, nb_of_bins).
        Is (np.ndarray):
            Frequency domain representation of the interference (nb_of_channels, nb_of_frames, nb_of_bins).
        threshold (float):
            Threshold in dB when comparing with interference (10 log(|Ts|**2) > 10 log(|Is|**2) + threshold).

    Returns:
        (np.ndarray):
            Ideal binary mask (nb_of_channels, nb_of_frames, nb_of_bins).
    """

    return (10.0 * np.log10(np.clip(np.abs(Ts) ** 2, a_min=1e-20, a_max=None)) > (10.0 * np.log10(np.clip(np.abs(Is) ** 2, a_min=1e-20, a_max=None)) + threshold))

