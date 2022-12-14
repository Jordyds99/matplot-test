# importing two required module
import numpy as np
import matplotlib.pyplot as plt 
import cv2

# Creating a numpy array
#X = np.array([1,2,3,4,5,5,5,5,5,4,3,2,1,1,1,1,1])
#Y = np.array([5,5,5,5,5,4,3,2,1,1,1,1,1,2,3,4,5])

# Creating equally spaced 100 data in range 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Setting radius
radius = 5

# Generating x and y data
X = radius * np.cos(theta)
Y = radius * np.sin(theta)

# Plotting point using scatter method
plt.scatter(X,Y)
plt.plot(X,Y)

#get current axes
ax = plt.gca()

#hide axes and borders
plt.axis('off')
#plt.show()
plt.savefig('plot.png', bbox_inches='tight')



im = cv2.imread('plot.png',cv2.IMREAD_COLOR)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im,contours,-1,(0,255,0),3)
cv2.imshow('output',im)
while True:
    if cv2.waitKey(6) & 0xff == 27:
        break