import numpy as np
import soundfile as sf


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

