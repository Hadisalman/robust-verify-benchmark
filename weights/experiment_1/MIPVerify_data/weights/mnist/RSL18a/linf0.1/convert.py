import numpy as np
import scipy.io as sio

"""
Converts weights provided by authors in `B1.npy`, `B2.npy`, `W1.npy` 
and `W2.npy` into a single `two-layer.mat` file.
"""

d = {}

b1 = np.load("B1.npy")
b2 = np.load("B2.npy")
w1 = np.load("W1.npy")
w2 = np.load("W2.npy")

d["fc1/weight"] = w1
d["fc1/bias"] = b1
d["logits/weight"] = w2
d["logits/bias"] = b2

sio.savemat("two-layer.mat", d)