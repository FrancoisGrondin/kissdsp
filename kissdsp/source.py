import numpy as np
import soundfile as sf


def read(file_name):
    """
    Read the audio file

    Args:
        file_name (string):
            The file name of the audio file.

    Returns:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    data, samplerate = sf.read(file_name, dtype=np.float32)

    return data.T


def white(nb_of_channels=1, nb_of_samples=90000):
    """
    Generate a non-coherent white noise source

    Args:
        nb_of_channels (int):
            Number of channels.
        nb_of_samples (int):
            Number of samples.

    Returns:
        (np.ndarray):
            Noise signal in the time domain (nb_of_channels, nb_of_samples).
    """
    return np.random.randn(nb_of_channels, nb_of_samples)

