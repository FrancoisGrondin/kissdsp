import numpy as np


def mvdr(vs, NNs=None):
    """
    Generate beamformer weights with MVDR. We compute the following equation:

    w(k) = phi_NN(k)^-1 d(k) / (d(k)^H phi_NN(k) d(k) )

    Args:
        vs (np.ndarray):
            The steering vector in the frequency domain (nb_of_bins, nb_of_channels).
        NNs (np.ndarray):
            The noise spatial covariance matrix (nb_of_bins, nb_of_channels, nb_of_channels). If set to None, then the
            noise spatial covariance matrix corresponds to identity matrices.

    Returns:
        (np.ndarray):
            The beamformer weights in the frequency domain (nb_of_bins, nb_of_channels).
    """

    nb_of_bins = vs.shape[0]
    nb_of_channels = vs.shape[1]

    if NNs is None:
        NNs = np.tile(np.expand_dims(np.identity(nb_of_channels), axis=0), reps=(nb_of_bins, 1, 1))

    NNsInv = np.linalg.inv(NNs)
    ds = np.expand_dims(vs, axis=2)
    dsH = np.conjugate(np.swapaxes(ds, -2, -1))
    ws = np.squeeze(NNsInv @ ds / np.tile(dsH @ NNsInv @ ds, reps=(1, nb_of_channels, 1)), axis=2)

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



