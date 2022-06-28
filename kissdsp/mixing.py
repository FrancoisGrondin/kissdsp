import numpy as np


def db(xs, levels):
    """
    Set the power in dB for each channel.

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
        levels (np.ndarray)
            Power level of each channel (nb_of_channels,)

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    ys = xs / np.expand_dims(np.sqrt(np.mean(xs ** 2, axis=1)), axis=1)
    ys *= 10 ** (np.expand_dims(levels, axis=1)/20)

    return ys


def vol(xs, level=1.0):
    """
    Normalize the volume to fit the signals in range [-1,+1].

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
        level (np.ndarray):
            Maximum amplitude for all channels (float)

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    ys = level * xs / np.max(np.abs(xs))

    return ys




