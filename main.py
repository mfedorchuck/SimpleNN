from PIL import Image

import glob
import numpy as np
import matplotlib.pyplot as plt

X = np.zeros((1,15))

# Here - check amount of files in "Dataset/" and put them into the 'X'
for filename in glob.glob('Dataset/*.bmp'):
    im = Image.open(filename).convert('LA')
    height, widht = im.size
    
    p = np.array(im)
    pict = np.reshape(p[:,:,0], height*widht)
    X = np.vstack([X, pict])

X = X[1:] / np.max(X);
#plt.matshow(X)# , fignum = 10, cmap = plt.cm.BuGn )
#plt.show

Y = np.zeros((100,10))
for val in range(10):
    Y[(val*10):(val*10 + 10), val] = 1;

# Sigmoid Function 
def nonlin(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# Initialise weights for NN
np.random.seed(1)

syn0 = 2*np.random.random((15, 10)) - 1
syn1 = 2*np.random.random((10, 10)) - 1 
                         
# Some gradient descent settings
iterations = 1500;

for iter in range(iterations):
    
    # Forward Propagation
    L0 = X
    L1 = nonlin(np.dot(L0, syn0))
    L2 = nonlin(np.dot(L1, syn1))
    
    #How bad are we now??
    L2_error = Y - L2
    L1_error = L2 - L1
    
    # Calculate how much we missed by slope of the sigmoid
    L2_delta = L2_error * nonlin(L2, True) * 1
    L1_delta = L1_error * nonlin(L1, True) 
    
    syn1 += np.dot(L1.T, L2_delta)
    syn0 += np.dot(L0.T, L1_delta) * 0.1

plt.matshow(L2_error)
plt.show()
print(np.max(L2_error))