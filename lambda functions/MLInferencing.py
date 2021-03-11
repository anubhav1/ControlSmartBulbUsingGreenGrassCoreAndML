
import logging
import os
import sys
from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2
from base64 import b64decode
import json
from spellchecker import SpellChecker

import greengrasssdk
from dlr import DLRModel

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Create MQTT client
mqtt_client = greengrasssdk.client('iot-data')


spell = SpellChecker()

# Initialize model
model_resource_path = os.environ.get('MODEL_PATH', '/neo-compiled-myhandwriting-model')
model = DLRModel(model_resource_path, 'cpu')

#logging.info(model)

def send_mqtt_message(message):
    logging.info("Publishing")
    mqtt_client.publish(topic='model/predictions', payload=message)
    
    

def predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Image", gray)
    #cv2.waitKey(0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #blur to reduce noise
    #cv2.imshow("Image", blurred)
    #cv2.waitKey(0)
    
    # perform edge detection
    edged = cv2.Canny(blurred, 30, 150)
    #cv2.imshow("Image", edged)
    #cv2.waitKey(0)
    
    #find contours of characters(objects) in image  and sort the
    #resulting contours from left-to-right
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if cnts:
        cnts = sort_contours(cnts, method="left-to-right")[0]

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the region of interest(roi) of character 
            roi = gray[y:y + h, x:x + w]
            
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

    # OCR the characters using our handwriting recognition model
    #preds = model.run(chars)
    numImages = chars.shape[0]
    preds = np.empty((0,18), dtype="float32")
    for i in range(numImages):
        pred = model.run(chars[[i],:])
        pred = pred[0]
        preds = np.append(preds, pred, axis=0)
    
    # define the list of label names
    labelNames = "ABCDEFGILMNOPRSTUV"
    #labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    readWord = ""
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        readWord += label


    readWord = spell.correction(readWord).upper()
    logging.info("************************************************************************************************")
    logging.info("Command  " + readWord)
    logging.info("************************************************************************************************")
    message = {}
    message['Command'] = readWord
    messageJson = json.dumps(message)
    send_mqtt_message(messageJson)



# The lambda to be invoked in Greengrass on topic handwritingmodel/frames
def lambda_handler(event, context):
    try:
        logging.info('Inferecing....')
        b64 = event
        JPEG = b64decode(b64["image"])
        image = cv2.imdecode(np.frombuffer(JPEG, dtype=np.uint8), cv2.IMREAD_COLOR)
        predict(image)
    except Exception as e:
        logger.exception(e)
        send_mqtt_message(
            'Exception occurred during prediction. Please check logs for troubleshooting: /greengrass/ggc/var/log.')
