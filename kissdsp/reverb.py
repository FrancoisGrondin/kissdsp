import numpy as np
import random as rnd
import rir_generator as rirgen


def room(mics, box, srcs, origin, alphas, c):
    """
    Generate room characteristics.

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
            Room info.
    """

    rm = {"mics": np.copy(mics),
          "box": np.copy(box),
          "srcs": np.copy(srcs),
          "origin": np.copy(origin),
          "alphas": np.copy(alphas),
          "c": c}

    return rm


def margin(rm):
    """
    Compute the margin relative to sources/microphones positioning.

    Args:
        rm (dict):
            Room info.

    Return:
        (float):
            Margin (in m) between microphones/srcs and walls/ceiling/floor ().
    """

    mics = rm["mics"] + np.expand_dims(rm["origin"], axis=0)
    srcs = rm["srcs"]

    margin_mics = np.zeros((mics.shape[0], 6), dtype=np.float32)
    margin_mics[:, 0] = np.abs(mics[:,0])
    margin_mics[:, 1] = np.abs(rm["box"][0] - mics[:,0])
    margin_mics[:, 2] = np.abs(mics[:,1])
    margin_mics[:, 3] = np.abs(rm["box"][1] - mics[:,1])
    margin_mics[:, 4] = np.abs(mics[:,2])
    margin_mics[:, 5] = np.abs(rm["box"][2] - mics[:,2])

    margin_srcs = np.zeros((srcs.shape[0], 6), dtype=np.float32)
    margin_srcs[:, 0] = np.abs(srcs[:,0])
    margin_srcs[:, 1] = np.abs(rm["box"][0] - srcs[:,0])
    margin_srcs[:, 2] = np.abs(srcs[:,1])
    margin_srcs[:, 3] = np.abs(rm["box"][1] - srcs[:,1])
    margin_srcs[:, 4] = np.abs(srcs[:,2])
    margin_srcs[:, 5] = np.abs(rm["box"][2] - srcs[:,2])

    margin = min([np.min(margin_mics), np.min(margin_srcs)])

    return margin


def distance(rm):
    """
    Compute the distances between sources and origin of microphone array.

    Args:
        rm (dict):
            Room info.

    Return:
        (np.ndarray):
            Distance (in m) between each source and the origin of the microphone array (nb_of_sources,).
    """

    srcs = rm["srcs"]
    micarray = rm["origin"]

    dist_srcs = srcs - np.expand_dims(micarray, axis=0)

    return dist_srcs


def thetas(rm):
    """
    Compute the angles between sources.

    Args:
        rm (dict):
            Room info.

    Return:
        (np.ndarray):
            Angles (in degrees) between all sources (nb_of_sources, nb_of_sources).
    """

    srcs = rm["srcs"]
    micarray = rm["origin"]    

    theta_srcs = np.zeros((srcs.shape[0], srcs.shape[0]), dtype=np.float32)
    
    for src_index1 in range(0, srcs.shape[0]):
        v1 = srcs[src_index1, :] - micarray
        for src_index2 in range(0, srcs.shape[0]):
            v2 = srcs[src_index2, :] - micarray
            theta_srcs[src_index1, src_index2] = (180.0/np.pi) * np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-20))

    return theta_srcs


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
            Convolved signals for each source (nb_of_channels, nb_of_samples)
    """

    nb_of_sources = hs.shape[0]
    nb_of_channels = hs.shape[1]
    nb_of_samples = ss.shape[1]
    rir_size = hs.shape[2]

    xs = np.zeros((nb_of_channels, nb_of_samples), dtype=np.float32)

    for source_index in range(0, nb_of_sources):
        for channel_index in range(0, nb_of_channels):
            xs[channel_index, :] += np.convolve(hs[source_index, channel_index, :], ss[source_index, :], mode='same')

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


def rotmat(yaw, pitch, roll):
    """
    Generate a rotation matrix with given yaw, pitch and roll

    Args:
        yaw (float):
            Yaw angle (in rad).
        pitch (float):
            Pitch angle (in rad).
        roll (float):
            Roll angle (in rad).

    Returns:
        (np.ndarray):
            The rotation matrix (3, 3).

    """

    R = np.zeros((3,3), dtype=np.float32)

    alpha = yaw
    beta = pitch
    gamma = roll    

    R[0, 0] = np.cos(beta) * np.cos(gamma)
    R[0, 1] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    R[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    R[1, 0] = np.cos(beta) * np.sin(gamma)
    R[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    R[1, 2] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    R[2, 0] = -np.sin(beta)
    R[2, 1] = np.sin(alpha) * np.cos(beta)
    R[2, 2] = np.cos(alpha) * np.cos(beta)

    return R