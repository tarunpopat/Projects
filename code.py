'''
Terms used:
nrc - no root canal needed.
rc - root canal needed.


If it gives [1] - root canal is needed.(positive detection for root canal)
If it gives [0] - no root canal needed.(negative detection for root canal)

'''
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np   
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
#these are the libraries


gray_img = cv2.imread('nrc1.png', cv2.IMREAD_GRAYSCALE)#grey 
cv2.imshow('nrc1',gray_img)
histnrc = cv2.calcHist([gray_img],[0],None,[256],[0,256])############ Making Histogram
#print(hist)
#plt.hist(gray_img.ravel(),256,[0,256])
#plt.title('Histogram for gray scale')
#plt.show()
#print(np.asarray(hist).shape)
histn2=np.asarray(histnrc[-50:])###### dimensionality Reduction (a form of PCA(Principle compound analysis))
   #print(histn2.shape)
   #print(histn2)
histn2=np.reshape(histn2,(50))###### dimensionality Reduction (a form of PCA(Principle compound analysis))



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
############# New Radiograph ########################


gray_imgrc = cv2.imread('rc1.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('rc1',gray_imgrc)
histrc = cv2.calcHist([gray_imgrc],[0],None,[256],[0,256])
#print(hist)
#plt.hist(gray_imgrc.ravel(),256,[0,256])
#plt.title('Histogram for gray scale rc')
#plt.show()
histr3=np.asarray(histrc[-50:])###### dimensionality Reduction (a form of PCA(Principle compound analysis))
histr3=np.reshape(histr3,(50))###### dimensionality Reduction (a form of PCA(Principle compound analysis))

#------------------------------------------------------------------------
#---------------------------------------------------------
############# New Radiograph ########################

gray_imgnrc2 = cv2.imread('nrc2.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('nrc2',gray_img)
histnrc2 = cv2.calcHist([gray_imgnrc2],[0],None,[256],[0,256])
#print(hist)
#plt.hist(gray_img.ravel(),256,[0,256])
#plt.title('Histogram for gray scale nrc')
#plt.show()
#print(np.asarray(hist).shape)
histn4=np.asarray(histnrc2[-50:])###### dimensionality Reduction (a form of PCA(Principle compound analysis))
histn4=np.reshape(histn4,(50))###### dimensionality Reduction (a form of PCA(Principle compound analysis))


#------------------------------------------------------------------------
#---------------------------------------------------------
############# New Radiograph ########################


gray_imgrc2 = cv2.imread('rc2.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('rc2',gray_imgrc)
histrc2 = cv2.calcHist([gray_imgrc2],[0],None,[256],[0,256])
#print(hist)
#plt.hist(gray_imgrc.ravel(),256,[0,256])
#plt.title('Histogram for gray scale rc')
#plt.show()
histr5=np.asarray(histrc2[-50:])###### dimensionality Reduction (a form of PCA(Principle compound analysis))
histr5=np.reshape(histr5,(50))###### dimensionality Reduction (a form of PCA(Principle compound analysis))


#print(hist2)
#print(hist2.shape)
X=np.asarray([histn2,histr3])
y=np.asarray([0,1])
#print(X)
#print(X.shape)

#######################  KNN Neighbours      #############################

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
#pred = neigh.predict(histn4)
print(neigh.predict(histn4))  ###### predicting output of hist4(histogram) image's diagnosis
