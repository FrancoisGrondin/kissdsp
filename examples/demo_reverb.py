import numpy as np

import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.visualize as vz

# Create a rectangular room with one source
rm = rb.room(mics=np.asarray([[-0.05, -0.05, +0.00], [-0.05, +0.05, +0.00], [+0.05, -0.05, +0.00], [+0.05, +0.05, +0.00]]),
             box=np.asarray([10.0, 10.0, 2.5]),
             srcs=np.asarray([[2.0, 3.0, 1.0]]),
             origin=np.asarray([4.0, 5.0, 1.25]),
             alphas=0.5 * np.ones(6),
             c=343.0)

# Generate room impulse responses
hs = rb.rir(rm)

# Split early and late reverberation
hse, hsl = rb.earlylate(hs)

# Display room impulse responses
vz.rir(hs)
vz.rir(hse)
vz.rir(hsl)