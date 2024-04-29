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

    # NN^-1 @ SSs
    numerator = np.linalg.solve(NNs, SSs)
    
    ws = (numerator / np.tile(np.expand_dims(np.trace(numerator, axis1=1, axis2=2), axis=(1,2)), reps=(1, nb_of_channels, nb_of_channels)))[:, :, 0]
    
    return ws


def gev(SSs, NNs):
    """
    Generate beamformer weights with GEV. We compute the following equation:

    w(k) = P{ phi_NN(k)^-1 phi_SS(k) }, where P{...} extracts the eigenvector with highest eigenvalue

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

    eigvalues, eigvectors = np.linalg.eigh(NNsInv @ SSs)
    ws = eigvectors[:, :, -1]

    return ws


def pfm(Xs, ws, phase_threshold=10):
    """
    Apply phase-based frequency masking.

    Args:
        Xs (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_frames, nb_of_bins).
        ws (np.ndarray):
            The steering vector in the frequency domain (nb_of_bins, nb_of_channels).
        phase_threshold (scalar):
            Phase difference threshold (in degrees) below which a frequency belongs to SOI.

    Returns:
        (np.ndarray):
            The time-frequency representation of both the SOI and the noise (2, nb_of_frames, nb_of_bins).
    """

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]
    nb_of_phasediffs = int((nb_of_channels*(nb_of_channels-1) / 2))

    # applying steering vector to align input channels towards SOI
    Xs = np.moveaxis(Xs, 0, -1)
    ws = np.tile(ws, reps=(nb_of_frames, 1, 1))
    Xs_aligned = np.conj(ws) * Xs

    # calculating average phase difference for each frequency bin
    Xs_phases = np.angle(Xs_aligned, deg=True)
    phase_diffs = np.zeros((nb_of_frames,nb_of_bins,nb_of_phasediffs))
    diff_i = 0
    for i in range(nb_of_channels):
        for j in range(i+1, nb_of_channels):
            phase_diffs[:,:,diff_i] = np.abs(Xs_phases[:,:,i] - Xs_phases[:,:,j])
            diff_i += 1
    phase_diffs[phase_diffs > 180] = np.abs(360-phase_diffs[phase_diffs > 180])
    phase_avgdiffs = np.mean(phase_diffs,axis=2)

    # masking reference microphone based on phase difference threshold to obtain:
    # - the SOI
    Ys = np.copy(Xs[:,:,0])
    Ys[phase_avgdiffs > phase_threshold] = 0.0
    # - the noise
    Is = np.copy(Xs[:,:,0])
    Is[phase_avgdiffs <= phase_threshold] = 0.0

    return np.concatenate((np.expand_dims(Ys,axis=0),np.expand_dims(Is,axis=0)),axis=0)


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



