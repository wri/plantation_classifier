import hickle as hkl
import numpy as np
import os

input_dir = "../../data/train-pytorch/"
band_count = 13
mins = np.full(band_count, np.inf)
maxs = np.full(band_count, -np.inf)

for fname in os.listdir(input_dir):
    if fname.endswith(".hkl"):
        x = hkl.load(os.path.join(input_dir, fname)).astype(np.float32)  # shape: (14,14,13)
        for b in range(band_count):
            band_vals = x[..., b]
            mins[b] = min(mins[b], np.min(band_vals))
            maxs[b] = max(maxs[b], np.max(band_vals))

print("min_all =", list(mins))
print("max_all =", list(maxs))
