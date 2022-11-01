import numpy as np


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


def scm(XXs):
    """
    Compute the Spatial Correlation Matrix

    Args:
        XXs (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins)

    Returns:
        (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)
    """

    Cs = np.transpose(np.mean(XXs, axis=2), (2, 0, 1))

    return Cs


def scmw(XXs, Ms):
    """
    Compute the Spatial Correlation Matrix using Ideal Ratio Masks

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

    MMs = np.tile(np.expand_dims(Ms, axis=0), (nb_of_channels, nb_of_channels, 1, 1))
    Cs = np.transpose(np.sum(XXs * MMs, axis=2) / np.sum(MMs, axis=2), (2, 0, 1))

    return Cs


def scmxw(XXs, Ms):
    """
    Compute the Spatial Correlation Matrices using Ideal Ratio Masks

    Args:
        XXs (np.ndarray):
            The cross spectrum (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins)
        Ms (np.ndarray)
            Ideal ratio mask (1, nb_of_frames, nb_of_bins)

    Returns:
        (np.ndarray):
            The spatial correlation matrix (nb_of_bins, nb_of_channels, nb_of_channels)
    """

    nb_of_channels = XXs.shape[0]
    nb_of_frames = XXs.shape[2]
    nb_of_bins = XXs.shape[3]

    MMs = np.tile(np.expand_dims(Ms, axis=0), (nb_of_channels, nb_of_channels, 1, 1))
    X2s = np.expand_dims(np.sum(np.abs(np.diagonal(XXs, axis1=0, axis2=1)), axis=2), axis=0)

    E_t_tt = np.sum(Ms * Ms * X2s, axis=(0, 1))
    E_t_ii = np.sum(Ms * (1-Ms) * X2s, axis=(0, 1))
    E_i_tt = np.sum((1-Ms) * Ms * X2s, axis=(0, 1))
    E_i_ii = np.sum((1-Ms) * (1-Ms) * X2s, axis=(0, 1))

    Phi_t = np.transpose(np.sum(MMs * XXs, axis=2), (2, 0, 1))
    Phi_i = np.transpose(np.sum((1-MMs) * XXs, axis=2), (2, 0, 1))

    E = np.zeros((nb_of_bins, 2, 2), dtype=np.float32)
    E[:, 0, 0] = E_t_tt
    E[:, 0, 1] = E_t_ii
    E[:, 1, 0] = E_i_tt
    E[:, 1, 1] = E_i_ii

    Einv = np.linalg.inv(E)

    a = np.tile(np.expand_dims(Einv[:, 0, 0], axis=(1,2)), (1, nb_of_channels, nb_of_channels))
    b = np.tile(np.expand_dims(Einv[:, 0, 1], axis=(1,2)), (1, nb_of_channels, nb_of_channels))
    c = np.tile(np.expand_dims(Einv[:, 1, 0], axis=(1,2)), (1, nb_of_channels, nb_of_channels))
    d = np.tile(np.expand_dims(Einv[:, 1, 1], axis=(1,2)), (1, nb_of_channels, nb_of_channels))

    phi_tt = a * Phi_t + b * Phi_i
    phi_ii = c * Phi_t + d * Phi_i

    E_tt = np.sum(Ms * X2s, axis=(0, 1))
    E_ii = np.sum((1-Ms) * X2s, axis=(0, 1))

    Phi_tt = np.tile(np.expand_dims(E_tt, axis=(1,2)), (nb_of_channels, nb_of_channels)) * phi_tt
    Phi_ii = np.tile(np.expand_dims(E_ii, axis=(1,2)), (nb_of_channels, nb_of_channels)) * phi_ii

    Cs = Phi_tt

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




