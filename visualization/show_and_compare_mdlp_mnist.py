# -*- coding: utf-8 -*-
"""
Shows image triplet original, discretized, and quantized for each example
in the data set. Close the previous image window to see the next one.

@author: Jussi Rasku
"""

#DATASET = "fashion"
DATASET = "mnist"
KIND = "train"

import mnist_reader
import numpy as np
import matplotlib.pyplot as plt

X_o, _ = mnist_reader.load_mnist('../data/%s'%DATASET, kind=KIND)

X_d, _ = mnist_reader.load_mnist('../data/%s_mdlp_discretized'%DATASET, kind=KIND)

X_q, _ = mnist_reader.load_mnist('../data/%s_mdlp_quantized'%DATASET, kind=KIND)

# Scale the bin indices to 0-255
max_bin_idx = np.max(X_d)
bin_scaler = 255/float(max_bin_idx)
for i in range(len(X_o)):
    oimg = X_o[i].reshape( (28,28) )
    dimg = (X_d[i].reshape( (28,28) )*bin_scaler).astype(np.uint8)
    qimg = X_q[i].reshape( (28,28) )
    f, axarr = plt.subplots(1,3)
    
    axarr[0].set_title("Original\nimage")
    axarr[0].imshow(oimg, vmin=0, vmax=255)
    axarr[1].set_title("MDLP\ndiscretized\nimage")
    axarr[1].imshow(dimg, vmin=0, vmax=255)
    axarr[2].set_title("MDLP\nbin-median-\nquantized\nimage")
    axarr[2].imshow(qimg, vmin=0, vmax=255)
    plt.show()