import numpy as np


def stft(xs, hop_size=128, frame_size=512):
    """
    Perform STFT

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
        hop_size (int):
            Space in samples between windows.
        frame_size (int):
            Number of samples per window.
    Returns:
        (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_frames, nb_of_bins)
    """

    nb_of_channels = xs.shape[0]
    nb_of_samples = xs.shape[1]
    nb_of_frames = int((nb_of_samples - frame_size + hop_size) / hop_size)
    nb_of_bins = int(frame_size/2+1)

    ws = np.tile(np.hanning(frame_size), (nb_of_channels, 1))
    Xs = np.zeros((nb_of_channels, nb_of_frames, nb_of_bins), dtype=np.csingle)

    for i in range(0, nb_of_frames):
        Xs[:, i, :] = np.fft.rfft(xs[:, (i*hop_size):(i*hop_size+frame_size)] * ws)

    return Xs


def istft(Xs, hop_size=128):
    """
    Perform iSTFT

    Args:
        Xs (np.ndarray):
            Signals in the frequency domain (nb_of_channels, nb_of_frames, nb_of_bins).
        hop_size (int):
            Space in samples between windows.
    Returns:
        (np.ndarray):
            The time-frequency representation (nb_of_channels, nb_of_samples).
    """

    pass


