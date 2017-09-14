from PIL import Image

import glob
import numpy as np
import matplotlib.pyplot as plt

# Here - check amount of files in "Dataset/" and put them into the 'X' array
X = np.zeros((1,15))
for filename in glob.glob('Dataset/*.bmp'):
    im = Image.open(filename).convert('LA')
    height, widht = im.size
    
    p = np.array(im)
    pict = np.reshape(p[:,:,0], height*widht)
    X = np.vstack([X, pict])

X = X[1:] / np.max(X);

Y = np.zeros((100,10))
for val in range(10):
    Y[(val*10):(val*10 + 10), val] = 1;

# Sigmoid Function 
def nonlin(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

iterations = 5000
# Initialise weights for the Network & array for plotting results
np.random.seed(113)
syn0 = 2*np.random.random((15, 10)) - 1
syn1 = 2*np.random.random((10, 10)) - 1 
PlotErr = np.zeros((iterations)) # 

for iter in range(iterations):
    # Forward Propagation
    L0 = X
    L1 = nonlin(np.dot(L0, syn0))
    L2 = nonlin(np.dot(L1, syn1))
    
    #How bad are we now??
    L2_error = Y - L2
    L1_error = L2 - L1
    PlotErr[iter] = np.max(abs(L2_error))
      
    # Calculate how much we missed by slope of the sigmoid
    L2_delta = L2_error * nonlin(L2, True) 
    L1_delta = L1_error * nonlin(L1, True) 
    
    syn1 += np.dot(L1.T, L2_delta * 1.2) 
    syn0 += np.dot(L0.T, L1_delta * 0.1)
    
## Displaying - how performance of network improving with amout of iterations
#plt.plot(PlotErr)
#plt.xlabel('number of iterations')
#plt.ylabel('peack errors in output layer')
#plt.title('Max differences between the output layer values and subscribed data')
#plt.show()

print('Max. error in the last (second) hidden layer after the training process: ')
print('(according to the training set)')
print(np.max(L2_error))

print()
print("Now 7 new images will be created and classified as numbers...")

#np.random.seed(seed = None)
fig = plt.figure(10)

def classification(InitIm, synap1, synap2):
    L0 = np.reshape(InitIm[:,:], 15)
    L1 = nonlin(np.dot(L0, synap1))
    return nonlin(np.dot(L1, synap2))

def IsValuable(RandIm):
    if np.max(RandIm) > 0.5:
        return 1
    return 0

def DigitOut(Num):
    if Num == 9:
        return 0
    return Num + 1

for index in range(7):
    IsVal = 0
    while (IsVal != 1):
        RandIm = abs(np.ceil(np.random.randn(5, 3) /10))
        OutPt = classification(RandIm, syn0, syn1)
        IsVal = IsValuable(OutPt)
        
    max_idx = np.argmax(OutPt)
    max_val = OutPt[max_idx]

    plt.subplot(7, 7, (index*7 + 1))
    plt.imshow(RandIm, cmap = plt.cm.gray)
    plt.axis('off') 
    
    plt.subplot(7, 7, (index*7 + 2))
    plt.axis('off')
    
    plt.text(0, 0.8, 'Identified as a symbol:')
    plt.text(0, 0.4, '"{}" with confidence: {:1.3}'.format(DigitOut(max_idx), max_val))
    
    OutPt[max_idx] = 0
    max_idx = np.argmax(OutPt)
    max_val = OutPt[max_idx]
    plt.text(0, 0.1, '"{}" with confidence: {:1.2}'.format(DigitOut(max_idx), max_val))
plt.show()