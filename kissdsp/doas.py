import numpy as np


def delay(doas, mics, c=343.0, sample_rate=16000):
    """
    Generate the time-difference of arrival (TDoA) between each mic and
    the center of mass of the array and the directions of arrival (in samples).

    Args:
        doas (np.ndarray):
            The directions of arrival (normalised magnitude) (nb_of_doas, 3).
        mics (np.ndarray):
            The microphone positions (nb_of_channels, 3).
        c (float):
            The speed of sound (in m/sec).
        sample_rate (int):
            The sample rate (in samples/sec).

    Returns:
        (np.ndarray):
            The TDoAs (nb_of_doas, nb_of_channels)
    """

    nb_of_doas = doas.shape[0]
    nb_of_channels = mics.shape[0]

    tdoas = np.zeros((nb_of_doas, nb_of_channels), dtype=np.float32)

    for channel_index in range(0, nb_of_channels):
        
        tdoas[:, channel_index] = (sample_rate/c) * doas @ mics[channel_index, :]
            
    return tdoas


def circle(points_count=360):
	"""
	Definition of a circular space (2D) for localization

	Args:
		points_count (scalar):
			The number of points to sample the circle.

	Returns:
		(np.ndarray):
			The circle (nb_of_points, 3).
	"""

	thetas = np.linspace(0, 2*np.pi, points_count+1)
	thetas = thetas[:-1]
	x = np.cos(thetas)
	y = np.sin(thetas)
	z = np.zeros(thetas.shape, dtype=np.float32)

	pts = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1), np.expand_dims(z,1)), axis=1)

	return pts


def sphere(levels_count=4):
	"""
	Definition of a spherical space (3D) for localization

	Args:
		levels_count (scalar):
			The number of levels to refine the icosahedron.

	Returns:
		(np.ndarray):
			The sphere (nb_of_points, 3).
	"""

	# Generate points at level 0

	h = np.sqrt(5.0) / 5.0
	r = (2.0/5.0) * np.sqrt(5.0)

	pts = np.zeros((12,3), dtype=float)
	pts[0,:] = [0,0,1]
	pts[11,:] = [0,0,-1]
	pts[np.arange(1,6,dtype=int),0] = r * np.sin(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(1,6,dtype=int),1] = r * np.cos(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(1,6,dtype=int),2] = h
	pts[np.arange(6,11,dtype=int),0] = -1.0 * r * np.sin(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(6,11,dtype=int),1] = -1.0 * r * np.cos(2.0 * np.pi * np.arange(0,5)/5.0)
	pts[np.arange(6,11,dtype=int),2] = -1.0 * h

	# Generate triangles at level 0

	trs = np.zeros((20,3), dtype=int)

	trs[0,:] = [0,2,1]
	trs[1,:] = [0,3,2]
	trs[2,:] = [0,4,3]
	trs[3,:] = [0,5,4]
	trs[4,:] = [0,1,5]

	trs[5,:] = [9,1,2]
	trs[6,:] = [10,2,3]
	trs[7,:] = [6,3,4]
	trs[8,:] = [7,4,5]
	trs[9,:] = [8,5,1]
	
	trs[10,:] = [4,7,6]
	trs[11,:] = [5,8,7]
	trs[12,:] = [1,9,8]
	trs[13,:] = [2,10,9]
	trs[14,:] = [3,6,10]
	
	trs[15,:] = [11,6,7]
	trs[16,:] = [11,7,8]
	trs[17,:] = [11,8,9]
	trs[18,:] = [11,9,10]
	trs[19,:] = [11,10,6]

	# Generate next levels

	for levels_index in range(0, levels_count):

		#      0
		#     / \
		#    A---B
		#   / \ / \
		#  1---C---2

		trs_count = trs.shape[0]
		subtrs_count = trs_count * 4

		subtrs = np.zeros((subtrs_count,6), dtype=int)

		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,0]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,0]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,0]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,1]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,2]
		subtrs[0*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,0]

		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,0]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,1]
		subtrs[1*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,2]

		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,2]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,0]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,1]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,2]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,2]
		subtrs[2*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,2]

		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),0] = trs[:,0]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),1] = trs[:,1]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),2] = trs[:,1]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),3] = trs[:,2]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),4] = trs[:,2]
		subtrs[3*trs_count+np.arange(0,trs_count,dtype=int),5] = trs[:,0]

		subtrs_flatten = np.concatenate((subtrs[:,[0,1]], subtrs[:,[2,3]], subtrs[:,[4,5]]), axis=0)
		subtrs_sorted = np.sort(subtrs_flatten, axis=1)

		unique_values, unique_indices, unique_inverse = np.unique(subtrs_sorted, return_index=True, return_inverse=True, axis=0)

		trs = np.transpose(np.reshape(unique_inverse, (3,-1)))

		pts = pts[unique_values[:,0],:] + pts[unique_values[:,1],:]
		pts /= np.repeat(np.expand_dims(np.sqrt(np.sum(np.power(pts,2.0), axis=1)), axis=1), 3, axis=1)

	return pts