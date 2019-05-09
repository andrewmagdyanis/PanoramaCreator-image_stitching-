'''
Discription:
	creating a panorma photo from 3 pictures
Authors:
	Andrew MAgdy Anis: andrewmagdyanis@gmail.com
	Amr Mohamed :
Date:
	8/5/2018
'''

# imports:
import imutils
import numpy as np
import cv2

# Loading images:
images =[]    #create a list for images
image1 =cv2.imread('1.jpg')
images.append(image1)
image2 =cv2.imread('2.jpg')
images.append(image2)
image3 =cv2.imread('3.jpg')
images.append(image3)

# stitching:
# initialize OpenCV's image sticher object and then perform the image
print("stitching images...")
stitcher = cv2.createStitcher(False)
(status, stitched) = stitcher.stitch(images)

# display and write the current result:
cv2.imwrite("Stitched1.jpg", stitched)
cv2.imshow("Stitched1", stitched)
cv2.waitKey(0)

# Checking if it done successfully: if the status is '0', then OpenCV successfully performed image
if status == 0:
	# create a 10 pixel border surrounding the stitched image
	print("cropping...")
	stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,cv2.BORDER_CONSTANT, (0, 0, 0))

	# display and write the current result:
	cv2.imwrite("Stitched2.jpg", stitched)
	cv2.imshow("Stitched2", stitched)
	cv2.waitKey(0)

	# convert the stitched image to grayscale and threshold it such that all pixels greater than zero are set to 255
	# (foreground) while all others remain 0 (background)
	gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

	# display and write the current result:
	cv2.imwrite("gray.jpg", gray)
	cv2.imshow("gray", gray)
	cv2.waitKey(0)

	# display and write the current result:
	cv2.imwrite("thresh.jpg", thresh)
	cv2.imshow("thresh", thresh)
	cv2.waitKey(0)

	# find all external contours in the threshold image
	# then find the *largest* contour which will be the contour/outline of the stitched image
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
	mask = np.zeros(thresh.shape, dtype="uint8")
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

	# display and write the current result:
	cv2.imwrite("mask.jpg",mask)
	cv2.imshow("mask", mask)
	cv2.waitKey(0)

	# create two copies of the mask: one to serve as our actual minimum rectangular region
	# and another to serve as a counter for how many pixels need to be removed to form the minimum rectangular region
	minRect = mask.copy()
	sub = mask.copy()

	# keep looping until there are no non-zero pixels left in the subtracted image
	while cv2.countNonZero(sub) > 0:
		# erode the minimum rectangular mask and then subtract the thresholded image from the minimum rectangular mask
		# so we can count if there are any non-zero pixels left
		cv2.imshow("minRect", minRect)
		cv2.waitKey(0)
		minRect = cv2.erode(minRect, None)
		sub = cv2.subtract(minRect, thresh)
		# display and write the current result:

	# find contours in the minimum rectangular mask and then extract the bounding box (x, y)-coordinates
	cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)

	# use the bounding box coordinates to extract the our final
	# stitched image
	stitched = stitched[y:y + h, x:x + w]

	# write the output stitched image to disk
	cv2.imwrite("output.jpg", stitched)

	# display the output stitched image to our screen
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints being detected
else:
	print("[INFO] image stitching failed ({})".format(status))