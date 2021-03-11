from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import csv


def processing(image):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image", gray)
    #cv2.waitKey(0)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #blur to reduce noise
    #cv2.imshow("Image", blurred)
    #cv2.waitKey(0)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150) #30, 15
    #cv2.imshow("Image", edged)
    #cv2.waitKey(0)

    #find contours of characters(objects) in images
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts:
        cnts = sort_contours(cnts, method="left-to-right")[0]
    # cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
    # cv2.imwrite("all_contours.jpg", image) 

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 180) and (h >= 15 and h <= 150): #180 #150
            # extract the region of interest(roi) of character 
            roi = gray[y:y + h, x:x + w]
            # cv2.imshow("Image", roi)
            # cv2.waitKey(0)

            # using a thresholding algorithm to make the character
            # appear as white (foreground) on a black background
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
            #cv2.imshow("Image", thresh)
            #cv2.waitKey(0)	

            # then grab the width and height of the thresholded image	
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)
            #cv2.imshow("Image", thresh)
            #cv2.waitKey(0)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))
            #cv2.imshow("Image", padded)
            #cv2.waitKey(0)
    
            padded = cv2.resize(padded, (32, 32))
            #cv2.imshow("Image", padded)
            #cv2.waitKey(0)  

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            #cv2.imshow("Image", padded)
            #cv2.waitKey(0)
  
            # update our list of characters(as padded images) that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    
    #writing characters in csv
    with open("myhandwriting.csv", "a+", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        list = chars.tolist()
        for listElement in list:
            row =[None]*1025
            row[0] = 22
            array = np.array(listElement, dtype="float32")*255.0
            array = array.astype(dtype='uint8') 
            array = np.squeeze(array, axis=2)
            #because our characters size is 32x32
            flat_array= array.reshape(1024,)
            flat_list = flat_array.tolist()
            row[1:] = flat_list
            writer.writerow(row)
        print("Character added into Dataset...")    

#Reading Images of characters for extracting those characters
cv2.destroyAllWindows()
frame = cv2.imread("V.png")
x = processing(frame)