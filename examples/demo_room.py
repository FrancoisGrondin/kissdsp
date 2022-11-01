import numpy as np

import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.visualize as vz

# Create a rectangular room with one source
rm = rb.room(mics=ma.minidsp_uma(),
             box=np.asarray([3.0, 3.0, 2.5]),
             srcs=np.asarray([[1.0, 2.0, 1.0], [2.0, 1.0, 1.5]]),
             origin=np.asarray([1.5, 1.5, 1.25]),
             alphas=0.5 * np.ones(6),
             c=343.0)

# Display room configuration
vz.room(rm)