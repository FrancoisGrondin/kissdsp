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

    if (len(data.shape) == 1):
        data = np.expand_dims(data, 1)

    return data.T


def write(xs, file_name):
    """
    Write the audio file

    Args:
        (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
        file_name (string):
            The file name of the audio file.
    """

    sf.write(file_name, xs.T, 16000)




