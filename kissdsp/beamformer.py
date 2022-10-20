import numpy as np


def mvdr(SSs, NNs):
    """
    Generate beamformer weights with MVDR. We compute the following equation:

    w(k) = ( phi_NN(k)^-1 phi_SS(k) / trace{phi_NN(k)^-1 phi_SS(k)} ) u

    Args:
        SSs (np.ndarray):
            The speech spatial covariance matrix (nb_of_bins, nb_of_channels, nb_of_channels).
        NNs (np.ndarray):
            The noise spatial covariance matrix (nb_of_bins, nb_of_channels, nb_of_channels).

    Returns:
        (np.ndarray):
            The beamformer weights in the frequency domain (nb_of_bins, nb_of_channels).
    """

    nb_of_bins = SSs.shape[0]
    nb_of_channels = SSs.shape[1]

    NNsInv = np.linalg.inv(NNs)
    
    ws = (NNsInv @ SSs / np.tile(np.expand_dims(np.trace(NNsInv @ SSs, axis1=1, axis2=2), axis=(1,2)), reps=(1, nb_of_channels, nb_of_channels)))[:, :, 0]
    
    return ws


def beam(Xs, ws):
    """
    Apply beamformer weights.

    Args:
        Xs (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_frames, nb_of_bins).
        ws (np.ndarray):
            The beamformer weights in the frequency domain (nb_of_bins, nb_of_channels).

    Returns:
        (np.ndarray):
            The time-frequency representation (1, nb_of_frames, nb_of_bins).
    """

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]

    Xs = np.expand_dims(np.moveaxis(Xs, 0, -1), axis=3)
    ws = np.tile(np.expand_dims(np.expand_dims(ws, axis=1), axis=0), reps=(nb_of_frames, 1, 1, 1))
    Ys = np.expand_dims(np.squeeze(np.squeeze(np.conj(ws) @ Xs, axis=-1), axis=-1), axis=0)

    return Ys


def avgpwr(Xs, ws):
    """
    Perform average of the power of each component of the beamformer.

    Args:
        Xs (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_frames, nb_of_bins).
        ws (np.ndarray):
            The beamformer weights in the frequency domain (nb_of_bins, nb_of_channels).

    Returns:
        (np.ndarray):
            The time-frequency square root power representation (1, nb_of_frames, nb_of_bins).
    """

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]

    Xs = np.moveaxis(Xs, 0, -1)
    ws = np.tile(np.expand_dims(ws, axis=0), reps=(nb_of_frames, 1, 1))
    Ys = np.expand_dims(np.sqrt(np.sum(np.abs(Xs * ws) ** 2, axis=2)), axis=0)

    return Ys



