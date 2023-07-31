import numpy as np

def si_sdr(s, s_hat):

	alpha = np.sum(s * s_hat) / np.sum(s ** 2)

	e_target = alpha * s
	e_res = s_hat - e_target

	ratio = 10 * np.log10(np.sum(e_target ** 2) / np.sum(e_res ** 2))

	return ratio