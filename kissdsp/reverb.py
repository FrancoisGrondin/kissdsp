import numpy as np
import random as rnd
import rir_generator as rirgen


def room(mics, box, srcs, origin, alphas, c):
    """
    Generate room characteristics

    Args:
        mics (np.ndarray):
            Microphones position in meters in the array as (x,y,z) coordinates (nb_of_channels, 3).
        box (np.ndarray):
            Room dimensions in meters as (x,y,z) dimensions (3,).
        srcs (np.ndarray):
            Sources positions in the room in meters as (x,y,z) coordinates (nb_of_sources, 3).
        origin (np.ndarray):
            Microphone array position in the room in meters as (x,y,z) coordinates (3,).
        alphas (np.ndarray):
            Absorption coefficients between ]0,1] (6,).
        c (float):
            Speed of sound (in m/sec) (1,).

    Returns:
        (dict):
            Room info
    """

    rm = {"mics": np.copy(mics),
          "box": np.copy(box),
          "srcs": np.copy(srcs),
          "origin": np.copy(origin),
          "alphas": np.copy(alphas),
          "c": c}

    return rm


def rndroom(mics, box_range, alpha_range, c_range, src_range, dists_range, margin):
    """
    Generate a random room configuration

    Args:
        mics (np.ndarray)
            Microphones position inmeter in the array as (x,y,z) coordinates (nb_of_channels, 3).
        box_range (np.ndarray)
            Min and max dimensions of the room in meters as (x_min, x_max, y_min, y_max, z_min, z_max) (6,).
        alpha_range (np.ndarray)
            Min and max values for the absorption coefficients (alpha_min, alpha_max) (2,).
        c_range (np.ndarray)
            Min and max values for the speed of sound (in m/sec) (2,).
        src_range (np.ndarray)
            Min and max numbers of sources (2,).
        dists_range (np.ndarray)
            Min and max distances in meter between each source and origin of the array (2,).
        margin (float)
            Minimum space in meter between walls, ceiling and floor and origin of the mics array or sources (1,).        
    
    Returns:
        (dict):
            Room info
    """

    # Set room size
    box = np.zeros(3)
    box[0] = rnd.uniform(box_range[0], box_range[1])
    box[1] = rnd.uniform(box_range[2], box_range[3])
    box[2] = rnd.uniform(box_range[4], box_range[5])    

    # Set mics origin
    origin = np.zeros(3)
    origin[0] = rnd.uniform(margin, box[0]-margin)
    origin[1] = rnd.uniform(margin, box[1]-margin)
    origin[2] = rnd.uniform(margin, box[2]-margin)

    # Set alpha value
    alphas = np.ones(6) * rnd.uniform(alpha_range[0], alpha_range[1])

    # Set c value
    c = rnd.uniform(c_range[0], c_range[1])

    # Set the number of sources
    srcs_count = rnd.randint(src_range[0], src_range[1])

    # Generate sources until distance and angle restrictions are met
    while True:

        # Generate random sources positions
        srcs = np.zeros((srcs_count, 3), dtype=np.float32)
        srcs[:, 0] = np.random.uniform(margin, box[0]-margin, size=srcs_count)
        srcs[:, 1] = np.random.uniform(margin, box[1]-margin, size=srcs_count)
        srcs[:, 2] = np.random.uniform(margin, box[2]-margin, size=srcs_count)

        # Compute distances
        dists = np.sqrt(np.sum((srcs - np.tile(np.expand_dims(origin, axis=0), (srcs_count, 1))) ** 2, axis=1))

        if (np.min(dists) > dists_range[0]) and (np.max(dists) < dists_range[1]):
            break

    # Create room dict
    rm = {"mics": np.copy(mics),
          "box": np.copy(box),
          "srcs": np.copy(srcs),
          "origin": np.copy(origin),
          "alphas": np.copy(alphas),
          "c": c}

    return rm


def rir(rm, sample_rate=16000, rir_size=4096):
    """
    Generate the room impulse response

    Args:
        rm (dict):
            Room characteristics.
        sample_rate (int):
            Sample rate (in samples/sec).
        rir_size (int):
            Number of coefficient for each impulse response.

    Return:
        (np.ndarray)
            Room impulse responses (nb_of_sources, nb_of_channels, rir_size).
    """

    nb_of_sources = rm["srcs"].shape[0]
    nb_of_channels = rm["mics"].shape[0]

    hs = np.zeros((nb_of_sources, nb_of_channels, rir_size))

    for source_index in range(0, nb_of_sources):

        h = rirgen.generate(c=rm["c"],
                            fs=sample_rate,
                            r=rm["mics"]+rm["origin"],
                            s=rm["srcs"][source_index, :],
                            L=rm["box"],
                            beta=np.sqrt(1.0 - rm["alphas"] ** 2),
                            nsample=rir_size)

        hs[source_index, :, :] = h.T

    return hs


def earlylate(hs, sample_rate=16000, early=0.05):
    """
    Split early reflection from late reverberation given the room impulse responses

    Args:
        hs (np.ndarray):
            Room impulse responses (nb_of_sources, nb_of_channels, rir_size).
        sample_rate (int):
            Sample rate (in samples/sec).            
        early (float):
            Early reflection time (in seconds).

    Return:
        (np.ndarray)
            Room impulse responses for early and late reverberation (nb_of_sources, nb_of_channels, rir_size).
    """

    nb_of_sources = hs.shape[0]
    nb_of_channels = hs.shape[1]
    rir_size = hs.shape[2]

    es = np.zeros((nb_of_sources, nb_of_channels, rir_size), dtype=np.float32)
    ls = np.zeros((nb_of_sources, nb_of_channels, rir_size), dtype=np.float32)

    for source_index in range(0, nb_of_sources):

        dp = np.amin(np.argmax(hs[source_index, :, :], axis=1))
        split = dp + int(early * sample_rate)

        es[source_index, :, :split] = hs[source_index, :, :split]
        ls[source_index, :, (split+1):] = hs[source_index, :, (split+1):]

    return es, ls


def conv(hs, ss):
    """
    Apply the room impulse responses to source signals

    Args:
        hs (np.ndarray):
            Room impulse responses (nb_of_sources, nb_of_channels, rir_size).
        ss (np.ndarray):
            Signals in the time domain (nb_of_sources, nb_of_samples).

    Return:
        (np.ndarray)
            Convolved signals for each source (nb_of_channels, nb_of_samples + rir_size - 1)
    """

    nb_of_sources = hs.shape[0]
    nb_of_channels = hs.shape[1]
    nb_of_samples = ss.shape[1]
    rir_size = hs.shape[2]

    xs = np.zeros((nb_of_channels, nb_of_samples + rir_size - 1), dtype=np.float32)

    for source_index in range(0, nb_of_sources):
        for channel_index in range(0, nb_of_channels):
            xs[channel_index, :] += np.convolve(hs[source_index, channel_index, :], ss[source_index, :])

    return xs


def doa(rm):
    """
    Extract the direction of arrival for each source

    Args:
        rm (dict):
            Room characteristics.

    Returns:
        (np.ndarray):
            Direction of arrival of sound, with (x,y,z) vector normalized to a magnitude of 1 (nb_of_sources, 3).
    """

    nb_of_sources = rm["srcs"].shape[0]

    doas = np.zeros((nb_of_sources, 3), dtype=np.float32)

    for source_index in range(0, nb_of_sources):
        doas[source_index, :] = rm["srcs"][source_index, :] - rm["origin"]
        doas[source_index, :] /= np.sqrt(np.sum(doas[source_index, :]**2))

    return doas


def tdoa(rm, sample_rate=16000):
    """
    Extract the time difference of arrival for each source (in sec)

    Args:
        rm (dict):
            Room characteristics.
        sample_rate (int):
            Sample rate (samples/sec).

    Returns:
        (np.ndarray):
            Time difference of arrival of sound (nb_of_sources, nb_of_channels).
    """

    nb_of_sources = rm["srcs"].shape[0]
    nb_of_channels = rm["mics"].shape[0]

    tdoas = np.zeros((nb_of_sources, nb_of_channels), dtype=np.float32)

    for source_index in range(0, nb_of_sources):
        dist = np.sqrt(np.sum((rm["srcs"][source_index, :] - (rm["origin"] + rm["mics"])) ** 2, axis=1))
        tdoas[source_index, :] = (sample_rate/rm["c"]) * dist
        tdoas[source_index, :] -= tdoas[source_index, 0]

    return tdoas

