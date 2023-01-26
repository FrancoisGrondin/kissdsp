import matplotlib.pyplot as plt
import numpy as np


def wave(xs):
    """
    Display waveform

    Args:
        xs (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_samples).
    """

    nb_of_channels = xs.shape[0]
    nb_of_samples = xs.shape[1]

    for channel_index in range(0, nb_of_channels):
        plt.subplot(nb_of_channels, 1, channel_index + 1)
        plt.plot(xs[channel_index, :])
    plt.show()


def spex(Xs):
    """
    Display spectrogram

    Args:
        Xs (np.ndarray):
            Signals in the frequency domain (nb_of_channels, nb_of_frames, nb_of_bins).
    """

    nb_of_channels = Xs.shape[0]
    nb_of_frames = Xs.shape[1]
    nb_of_bins = Xs.shape[2]

    for channel_index in range(0, nb_of_channels):
        plt.subplot(nb_of_channels, 1, channel_index + 1)
        plt.imshow(np.log(np.abs(Xs[channel_index, :, :]) + 1e-10).T, aspect='auto', origin='lower')
    plt.show()


def phase(XXs):
    """
    Display the phase between each pair

    Args:
        XXs (np.ndarray):
            Signals in the frequency domain (nb_of_channels, nb_of_channels, nb_of_frames, nb_of_bins).
    """

    nb_of_channels1 = XXs.shape[0]
    nb_of_channels2 = XXs.shape[1]
    nb_of_frames = XXs.shape[2]
    nb_of_bins = XXs.shape[3]

    pair_index = 0
    for channel_index1 in range(0, nb_of_channels1):
        for channel_index2 in range(0, nb_of_channels2):
            pair_index += 1
            plt.subplot(nb_of_channels1, nb_of_channels2, pair_index)
            plt.imshow((np.angle(XXs[channel_index1, channel_index2, :, :])).T,
                       aspect='auto', origin='lower')
    plt.show()


def mask(Ms):
    """
    Display the Ideal Ratio or Binary Masks

    Args:
         Ms (np.ndarray):
            Masks in the frequency domain (nb_of_channels, nb_of_frames, nb_of_bins).
    """

    nb_of_channels = Ms.shape[0]
    nb_of_frames = Ms.shape[1]
    nb_of_bins = Ms.shape[2]

    for channel_index in range(0, nb_of_channels):
        plt.subplot(nb_of_channels, 1, channel_index + 1)
        plt.imshow(Ms[channel_index, :, :].T, aspect='auto', origin='lower')
    plt.show()


def beampattern(vs):
    """
    Display the beampattern

    Args:
        vs (np.ndarray):
            Steering or mixing vector in the frequency domain (nb_of_bins, nb_of_channels).
    """

    nb_of_bins = vs.shape[0]
    nb_of_channels = vs.shape[1]

    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(vs), aspect='auto', origin='lower')
    plt.subplot(2, 1, 2)
    plt.imshow(np.angle(vs), aspect='auto', origin='lower')
    plt.show()


def rir(hs):
    """
    Display the room impulse responses

    Args:
        hs (np.ndarray):
            Room impulse responses (nb_of_sources, nb_of_channels, nb_of_samples)
    """

    nb_of_sources = hs.shape[0]
    nb_of_channels = hs.shape[1]
    nb_of_samples = hs.shape[2]

    for source_index in range(0, nb_of_sources):
        plt.subplot(nb_of_sources, 1, source_index+1)
        plt.plot(hs[source_index, :, :].T)

    plt.show()


def room(rm):
    """
    Display the room geometry

    Args:
        rm (dict):
            Room info
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(rm['mics'][:, 0] + rm['origin'][0], 
               rm['mics'][:, 1] + rm['origin'][1], 
               rm['mics'][:, 2] + rm['origin'][2],
               color='b')
    ax.scatter(rm['srcs'][:, 0],
               rm['srcs'][:, 1],
               rm['srcs'][:, 2],
               color='r')

    ax.set_xlim([0, rm['box'][0]])
    ax.set_ylim([0, rm['box'][1]])
    ax.set_zlim([0, rm['box'][2]])

    plt.show()


def xcorr(xxs):
    """
    Display the cross correlation

    Args:
        xxs (np.ndarray):
            Signals in the time domain (nb_of_channels, nb_of_channels, nb_of_frames, frame_size).
    """

    nb_of_channels1 = xxs.shape[0]
    nb_of_channels2 = xxs.shape[1]
    nb_of_frames = xxs.shape[2]
    frame_size = xxs.shape[3]

    pair_index = 0
    for channel_index1 in range(0, nb_of_channels1):
        for channel_index2 in range(0, nb_of_channels2):
            pair_index += 1
            plt.subplot(nb_of_channels1, nb_of_channels2, pair_index)
            plt.imshow(xxs[channel_index1, channel_index2, :, :].T,
                       aspect='auto', origin='lower')
    plt.show()



    
