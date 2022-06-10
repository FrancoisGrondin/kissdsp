import numpy as np


def source(xs, source_index):
    """
    Extract specific source

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_sources, nb_of_channels, nb_of_samples).

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    return xs[source_index, :, :]


def channel(xs, channel_index):
    """
    Extract specific channel

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_sources, nb_of_channels, nb_of_samples).

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    return xs[:, channel_index, :]


def collapse(xs):
    """
    Combine all sources together

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_sources, nb_of_channels, nb_of_samples).

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    return np.sum(xs, axis=0)


def concatenate(xs1, xs2):
    """
    Concatenate sources together

    Args:
        xs1 (np.ndarray):
            Signals in the time domain (nb_of_sources1, nb_of_channels, nb_of_samples).
        xs2 (np.ndarray):
            Signals in the time domain (nb_of_sources2, nb_of_channels, nb_of_samples).

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_sources1 + nb_of_sources2, nb_of_channels, nb_of_samples).
    """

    return np.concatenate((xs1, xs2), axis=0)


