# -*- coding: utf-8 -*-
"""
Discretize the mnist/fashion data with the MDLP algorithm implementation 
from navicto (inclided into this project as a git submodule).

@author: Jussi Rasku
"""

import numpy as np
import pickle
from collections import defaultdict

from Discretization-MDLPC.MDLP import MDLP_Discretizer

READ_CACHED = True
DATASET = "fashion" #"mnist"
DATASET = "mnist"


####################################
# 1. Read the MNIST Fashion dataset
####################################

import mnist_reader
X_train, y_train = mnist_reader.load_mnist('../data/'+DATASET, kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/'+DATASET, kind='t10k')

if READ_CACHED:
    X_train_discretized = pickle.load( open("train_discretized.p","rb") )
    X_test_discretized = pickle.load( open("test_discretized.p","rb") )
    pix_bin_grayscale_mapping = pickle.load( open("discretized_bin_to_grayscale_map.p","rb") )
else:
    
    ###################
    # 2. Discretize it
    ###################
    
    
    pixel_idx_array = np.arange(X_train.shape[1])
    discretizer = MDLP_Discretizer(features=pixel_idx_array)
    discretizer.fit(X_train, y_train)
    
    X_train_discretized = discretizer.transform(X_train).astype(np.uint8)
    X_test_discretized = discretizer.transform(X_test).astype(np.uint8)
    
    # Pickle the results
    pickle.dump( X_train_discretized, open( "train_discretized.p", "wb" ) )
    pickle.dump( X_test_discretized, open( "test_discretized.p", "wb" ) )

    ################################
    # 3. Build quantization mapping
    ################################
    
    # Create a mapping from the bin number to grayscale value (per pixel index)
    # TODO: Two for loops -> O(n^2) b/c my Numpy-fu failed me. However, this will
    # get the job done for now. We will cache the results and consider rewriting
    # this later (yeah right :).
    pix_bin_grayscale_mapping = defaultdict(int)
    nbins_per_pix = np.max(X_train_discretized, axis=0)+1
    for pix_idx, nbins in enumerate(nbins_per_pix):
        print("Building the pixel to bin map for pixel %d (%d bins)"%(pix_idx,nbins) )
        binned_pix = X_train_discretized[:,pix_idx]
        prev_grayscale_value = -1
        for bin_idx in range(nbins):
            grayscale_value = int(np.median( X_train[ binned_pix==bin_idx, pix_idx ] ))
            pix_bin_grayscale_mapping[(pix_idx, bin_idx)]=grayscale_value
            if grayscale_value==prev_grayscale_value:
                print("WARNING: two bins have the same average grayscale value")
            prev_grayscale_value = grayscale_value
            
    pickle.dump( pix_bin_grayscale_mapping, open( "discretized_bin_to_grayscale_map.p", "wb" ) )
 
print("Discretized train feature vector size", X_train_discretized.shape)
print("Discretized test feature vector size", X_test_discretized.shape)

# Try to remove the features with only one bin (not enough entropy to
#  contribute to the classification task).
# fashion -> No need, only the first pixel has no information.
one_class_features = np.any(X_train_discretized, axis=0)
X_train_discretized_dr = X_train_discretized[:,one_class_features]
X_test_discretized_dr = X_test_discretized[:,one_class_features]
print("Discretized DR feature vector size", X_train_discretized_dr.shape) 


###########################
# 4. Save in mnist format
###########################

import gzip
def save_mninst_image_gzipfile(filebasename, data, header):
    with gzip.open(filebasename, 'wb') as f:
        f.write(header)
        f.write(data.flatten().tobytes())
    
save_mninst_image_gzipfile("train-discretized-images-idx3-ubyte.gz",
                           X_train_discretized,
                           # this magic string is the 16 first bytes of the uncompressed mnist train data
                           b'\x00\x00\x08\x03\x00\x00\xea`\x00\x00\x00\x1c\x00\x00\x00\x1c')
save_mninst_image_gzipfile("t10k-discretized-images-idx3-ubyte.gz",
                           X_test_discretized,
                           # this magic string is the 16 first bytes of the uncompressed mnist test data
                           b"\x00\x00\x08\x03\x00\x00'\x10\x00\x00\x00\x1c\x00\x00\x00\x1c")

def discretized_to_grayscale(fvector, valuemap, size=(28,28)):
    """ Simple convinience function to convert discretized image back to 
    grayscale using the provided grayscale bin to value mapping.
    """
    return np.array( [int(valuemap[kv]) for kv in enumerate(fvector)] ).reshape(size)

X_train_quantized = np.apply_along_axis(
    lambda x: discretized_to_grayscale(x, pix_bin_grayscale_mapping, x.shape),
    1, X_train_discretized).astype(np.uint8)
X_test_quantized = np.apply_along_axis(
    lambda x: discretized_to_grayscale(x, pix_bin_grayscale_mapping, x.shape),
    1, X_test_discretized).astype(np.uint8)

print("Quantized train feature vector size", X_train_quantized.shape)
print("Quantized test feature vector size", X_test_quantized.shape)

save_mninst_image_gzipfile("train-quantized-images-idx3-ubyte.gz",
                           X_train_quantized,
                           b'\x00\x00\x08\x03\x00\x00\xea`\x00\x00\x00\x1c\x00\x00\x00\x1c')
save_mninst_image_gzipfile("t10k-quantized-images-idx3-ubyte.gz",
                           X_test_quantized,
                           b"\x00\x00\x08\x03\x00\x00'\x10\x00\x00\x00\x1c\x00\x00\x00\x1c")
  
print("DONE. Exported the transformed data .gz files")
