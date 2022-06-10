import numpy as np


def scm(Xs, Ms=None):
    """
    Compute the Spatial Correlation Matrix

    Args:
        Xs (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_frames, nb_of_bins)
        Ms (np.ndarray):
            The time-frequency mask (nb_of_channels, nb_of_frames, nb_of_bins). If set to None, then mask is all 1's.

    Returns:
        (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)
    """

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]


def steering(XXs):
    """
    Compute the Steering Vector (rank 1)

    Args:
        XXs (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)

    Returns:
        (np.ndarray):
            The steering vector in the frequency domain (nb_of_bins, nb_of_channels)
    """

    nb_of_bins = XXs.shape[0]
    nb_of_channels = XXs.shape[1]

    vs = np.linalg.eigh(XXs)[1][:, :, -1]
    v0s = np.tile(np.expand_dims(vs[:, 0], axis=1), (1, nb_of_channels))
    vs /= np.exp(1j * np.angle(v0s))

    return vs


def freefield(tdoas, frame_size=512):
    """
    Generate the Free Field Spatial Correlation Matrix (rank 1)

    Args:
        tdoas (np.ndarray):
            The time differences of arrival for each channel (nb_of_channels).
    frame_size (int):
        Number of samples per window.

    Returns:
        (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels).
    """

    nb_of_channels = len(tdoas)
    nb_of_bins = int(frame_size/2)+1

    ks = np.arange(0, nb_of_bins)
    As = np.exp(-1j * 2.0 * np.pi * np.expand_dims(ks, axis=1) @ np.expand_dims(tdoas, axis=0) / frame_size)
    AAs = np.expand_dims(As, axis=2) @ np.conj(np.expand_dims(As, axis=1))

    return AAs

