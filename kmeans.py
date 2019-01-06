import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import IncrementalPCA, PCA
from os import listdir, mkdir
from os.path import isfile, join, splitext, exists
from PIL import Image
import numpy
from matplotlib.image import imread


N_COLORS = 56
MAX_SAMPLES = 2000
GRAY_IMAGE = False
RESIZE_PERCENT = 0


DIRECTORY_NAME = input("Directory name ? ")
N_COLORS = int(input("Number of colors ? "))

cimage = str(input("Coloured image(y/n) ? "))
RESIZE_PERCENT = float(input("Percent resize")) 

GRAY_IMAGE = True
if  cimage == 'y':
    GRAY_IMAGE = False


# use jpg files
onlyfiles = [f for f in listdir(DIRECTORY_NAME) if isfile(join(DIRECTORY_NAME, f))]
onlyfiles = [f for f in  onlyfiles if str(f).endswith("jpg")]

    
for file in onlyfiles:
    print(file)
    sourceFileName = join(DIRECTORY_NAME, file)
    img = imread(sourceFileName)/255
    plt.imshow(img)
    
    w, h, d = original_shape = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:MAX_SAMPLES]
    
    kmeans = KMeans(n_clusters=N_COLORS, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    clusters = kmeans.cluster_centers_
    
    result = np.zeros(shape = (len(labels),3), dtype = np.float64)
    for i in range(len(labels)):
        result[i] = clusters[labels[i]]
        
    if GRAY_IMAGE:
        result = np.amax(result, axis = 1)
        result.shape = (-1,1)
        result = np.round(result,3)
 
        print("shades of gray " + str(len(np.unique(result))))
        result = np.hstack((result, result, result))
        
    result = np.reshape(result, (w,h,d))
    plt.imshow(result)
    
    img = Image.fromarray((result * 255).astype(numpy.uint8))
    if not exists(DIRECTORY_NAME + "\Reduced"):
        mkdir(DIRECTORY_NAME + "\Reduced")
    
    newFileName = join(DIRECTORY_NAME + "\Reduced", splitext(file)[0]+"Reduced"+".jpg")
        
    if RESIZE_PERCENT < 100:         
        newW = int((RESIZE_PERCENT/100)*w)
        newH = int((RESIZE_PERCENT/100)*h)        
        img = img.resize((newH, newW), Image.LANCZOS)
        
    img.save(newFileName)
        
    

    
    

