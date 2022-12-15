# importing two required module
import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv

# Creating a numpy array
X = np.array([1,2,3,4,8,8,8,8,8,4,3,2,1,1,1,1,1,5,2,5,7,9,4,1,0])
Y = np.array([5,5,5,5,5,4,3,2,1,1,1,1,1,2,3,4,5,7,1,6,3,2,5,7,3])

# Creating equally spaced 100 data in range 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Setting radius
radius = 5

# Generating x and y data
#X = radius * np.cos(theta)
#Y = radius * np.sin(theta)

# Plotting point using scatter method
plt.scatter(X,Y)
plt.plot(X,Y)



#get current axes
ax = plt.gca()

#hide axes and borders
plt.axis('off')
#plt.show()
plt.savefig('plot.png', bbox_inches='tight')



# im = cv.imread('plot.png',cv.IMREAD_COLOR)
# imgray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
# ret,thresh = cv.threshold(imgray,127,255,cv.THRESH_BINARY)
# contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(im,contours,-1,(0,255,0),3)
# cv.imshow('output',im)
# while True:
#     if cv.waitKey(6) & 0xff == 27:
#         break



# change it with your absolute path for the image
image = cv.imread("plot.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5),
					cv.BORDER_DEFAULT)
ret, thresh = cv.threshold(blur, 200, 255,
						cv.THRESH_BINARY_INV)

#cv.imwrite("thresh.png",thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(thresh.shape[:2],
				dtype='uint8')

# cv.drawContours(blank, contours, -1,
# 				(250, 20, 150), 1)

#cv.imwrite("Contours.png", blank)


for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(image, [i], -1, (120, 255, 0), 2)
        cv.circle(image, (cx, cy), 7, (200, 140, 255), -1)
        cv.putText(image, "t", (cx - 20, cy - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    print(f"x: {cx} y: {cy}")

#cv.imwrite("image.png", image)


cv.imshow('dingen',image)
while True:
    if cv.waitKey():
        break

