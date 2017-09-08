
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

    # Here - check amount of files in "Dataset/"

im = Image.open("Dataset/009.bmp").convert('LA')
height, widht = im.size
   
p = np.array(im)
pict = np.asarray(p[:,:,0])


p2=[]
for row in range(height):
     for col in range(widht):
         a = im.getpixel((row,col))
         p2.append(a)
p3=np.asarray(p2)

plt.matshow(pict)# , fignum = 10, cmap = plt.cm.BuGn )
plt.show