# importing two required module
import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv

# Creating a numpy array
X = np.array([1,4,4,1,1,10,14,14,10,10])
Y = np.array([1,1,4,4,1,1,1,4,4,1])
# Circle
# theta = np.linspace(0, 2 * np.pi, 100)
# radius = 5
# X = radius * np.cos(theta)
# Y = radius * np.sin(theta)

# Plotting point using scatter method
plt.scatter(X,Y)
plt.plot(X,Y)
#hide axes and borders
ax = plt.gca()
plt.axis('off')
#plt.show()
plt.savefig('plot.png', bbox_inches='tight')

# change it with your absolute path for the image
image = cv.imread("plot.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5),
					cv.BORDER_DEFAULT)
ret, thresh = cv.threshold(blur, 200, 255,
						cv.THRESH_BINARY_INV)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(thresh.shape[:2],
				dtype='uint8')


for i in contours:
    M = cv.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.drawContours(image, [i], -1, (120, 255, 0), 2)
        cv.circle(image, (cx, cy), 7, (200, 140, 255), -1)
        cv.putText(image, "t", (cx - 20, cy - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


cv.imshow('dingen',image)
while True:
    if cv.waitKey():
        break
