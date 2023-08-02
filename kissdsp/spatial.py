import numpy as np

import matplotlib.pyplot as plt

def xspec(Xs):
    """
    Compute the Cross Spectrum

    Args:
        Xs (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_frames, nb_of_bins)

    Returns:
        (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins)
    """

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]

    XXs = np.zeros((nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins), dtype=np.csingle)

    for channel_index1 in range(0, nb_of_channels):
        
        for channel_index2 in range(0, nb_of_channels):

            XXs[channel_index1, channel_index2, :, :] = Xs[channel_index1, :, :] * np.conj(Xs[channel_index2, :, :])

    return XXs


def scm(XXs, Ms=None):
    """
    Compute the Spatial Correlation Matrix

    If mask is provided, weight with the mask. If not assume there is no mask.

    Args:
        XXs (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins)
        Ms (np.ndarray):
            Ideal ratio mask (1, nb_of_frames, nb_of_bins)            

    Returns:
        (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)
    """

    nb_of_channels = XXs.shape[0]
    nb_of_frames = XXs.shape[2]
    nb_of_bins = XXs.shape[3]

    if Ms is None:
        Ms = np.ones((1, nb_of_frames, nb_of_bins), dtype=np.float32)

    MMs = np.tile(np.expand_dims(Ms, axis=0), (nb_of_channels, nb_of_channels, 1, 1))
    Cs = np.transpose(np.sum(XXs * MMs, axis=2) / np.sum(MMs, axis=2), (2, 0, 1))

    return Cs


def oscm(XXs, Ms=None, alpha=0.1):
    """
    Compute the Online Spatial Correlation Matrix

    If mask is provided, weight with the mask. If not assume there is no mask.

    Args:
        XXs (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins)
        Ms (np.ndarray):
            Ideal ratio mask (1, nb_of_frames, nb_of_bins)
        alpha (float):
            Adaptation rate (between 0 and 1)

    Returns:
        (np.ndarray):
            The online spatial correlation matrix (nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels)
    """

    nb_of_channels = XXs.shape[0]
    nb_of_frames = XXs.shape[2]
    nb_of_bins = XXs.shape[3]

    if Ms is None:
        Ms = np.ones((1, nb_of_frames, nb_of_bins), dtype=np.float32)

    MMs = np.tile(np.expand_dims(Ms, axis=0), (nb_of_channels, nb_of_channels, 1, 1))

    XXMMs = XXs * MMs

    # Loop for each frame: not very efficient, should be improved
    Cs = np.zeros((nb_of_frames, nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)

    C = np.zeros((nb_of_bins, nb_of_channels, nb_of_channels), dtype=np.csingle)

    for frame_index in range(nb_of_frames):
        C *= (1 - alpha)
        C += alpha * np.transpose(XXMMs[:, :, frame_index, :], (2, 0, 1))
        Cs[frame_index, :, :, :] = C

    return Cs    


def steering(Cs):
    """
    Compute the Steering Vector (rank 1)

    Args:
        Cs (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)

    Returns:
        (np.ndarray):
            The steering vector in the frequency domain (nb_of_bins, nb_of_channels)
    """

    nb_of_bins = Cs.shape[0]
    nb_of_channels = Cs.shape[1]

    vs = np.linalg.eigh(Cs)[1][:, :, -1]
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


def diagload(Cs, gamma=0.01, epsilon=1e-20):
    """
    Load the spatial correlation matrix with a diagonal to avoid singularity

    Cs' = Cs + (gamma * trace(Cs) + epsilon) * I

    Args:
        Cs (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels).
        gamma (float):
            Gain applied to the trace of the spatial correlation matrix to set energy of the identity matrix.
        epsilon (float):
            Offset to add the identity matrix.

    Returns:
        (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels).
    """

    nb_of_bins = Cs.shape[0]
    nb_of_channels = Cs.shape[1]

    tr = np.expand_dims(np.expand_dims(np.trace(Cs, axis1=1, axis2=2), axis=1), axis=1)
    trs = np.tile(tr, (1, nb_of_channels, nb_of_channels))
    I = np.expand_dims(np.eye(nb_of_channels), axis=0)
    Is = np.tile(I, (nb_of_bins, 1, 1))

    Csp = Cs + (gamma * trs + epsilon) * Is

    return Csp










