import numpy as np


def irm(Ts, Rs):
    """
    Generate Ideal Ratio Mask

    Args:
        Ts (np.ndarray):
            Frequency domain representation of the target (nb_of_channels, nb_of_frames, nb_of_bins).
        Rs (np.ndarray):
            Frequency domain representation of the residual (nb_of_channels, nb_of_frames, nb_of_bins).

    Returns:
        (np.ndarray):
            Ideal ratio mask (nb_of_channels, nb_of_frames, nb_of_bins).
    """

    return np.clip((np.abs(Ts) ** 2) / (np.abs(Ts) ** 2 + np.abs(Rs) ** 2 + 1e-20), a_min=0.0, a_max=1.0)


def ibm(Ts, Rs, threshold=0.0):
    """
    Generate Ideal Binary Mask

    Args:
        Ts (np.ndarray):
            Frequency domain representation of the target (nb_of_channels, nb_of_frames, nb_of_bins).
        Rs (np.ndarray):
            Frequency domain representation of the residual (nb_of_channels, nb_of_frames, nb_of_bins).
        threshold (float):
            Threshold in dB when comparing with residual (10 log(|Ts|**2) > 10 log(|Rs|**2) + threshold).

    Returns:
        (np.ndarray):
            Ideal binary mask (nb_of_channels, nb_of_frames, nb_of_bins).
    """

    return (10.0 * np.log10(np.clip(np.abs(Ts) ** 2, a_min=1e-20, a_max=None)) > (10.0 * np.log10(np.clip(np.abs(Rs) ** 2, a_min=1e-20, a_max=None)) + threshold))


def mean(Ms):
    """
    Compute Mean Mask from Multiple Channels

    Args:
        Ms (np.ndarray):
            Time-frequency mask (nb_of_channels, nb_of_frames, nb_of_bins).

    Returns:
        (np.ndarray):
            Time-frequency mask (1, nb_of_frames, nb_of_bins).
    """

    return np.mean(Ms, axis=0, keepdims=True)


def median(Ms):
    """
    Compute Median Mask for Multiple Channels

    Args:
        Ms (np.ndarray):
            Time-frequency mask (nb_of_channels, nb_of_frames, nb_of_bins).

    Returns:
        (np.ndarray):
            Time-frequency mask (1, nb_of_frames, nb_of_bins).
    """

    return np.median(Ms, axis=0, keepdims=True)

    