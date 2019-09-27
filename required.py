#required packages
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
#Mounting drive to access dataset  
from google.colab import drive
drive.mount('/content/drive')
# This method to reduce 512 by 512 images to 256 by 256

def compress_image(original_image):
  size = (256, 256)
  fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)

  return np.array(fit_and_resized_image.getdata()).reshape(256,256)
from PIL import Image, ImageOps

input_X=[]
output_Y = []

for i in range(1,7):
    #fetching data
    
    INPUT_SCAN_FOLDER='/content/drive/My Drive/Class'+str(i) # path where the dataset is stored
    
    for dirName, subdirList, fileList in os.walk(INPUT_SCAN_FOLDER):
        count = 0    
        for filename in fileList:
            if(count >= 200):
                break
            count+=1
            if ".png" in filename.lower():
                    input_X.append(compress_image(Image.open(os.path.join(dirName, filename))))
                    output_Y.extend([0])
                    #print(len(input_X))
    print(i)
    INPUT_SCAN_FOLDER='/content/drive/My Drive/Class'+str(i)+'_def' # path where the dataset is stored
    
    for dirName, subdirList, fileList in os.walk(INPUT_SCAN_FOLDER):
            for filename in fileList:
                if ".png" in filename.lower():
                    input_X.append(compress_image(Image.open(os.path.join(dirName, filename))))
                    output_Y.extend([i])
#output:1 2 3 4 5 6
print(len(input_X))
#output:2103