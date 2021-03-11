import os
import sys
import time
import uuid
import json
import logging
import cv2
from base64 import b64encode, b64decode

from AWSIoTPythonSDK.core.greengrass.discovery.providers import DiscoveryInfoProvider
from AWSIoTPythonSDK.core.protocol.connection.cores import ProgressiveBackOffCore
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from AWSIoTPythonSDK.exception.AWSIoTExceptions import DiscoveryInvalidRequestException
from deviceConfigs import *

#General message notification callback
def customOnMessage(message):
    print('Received message on topic %s: %s\n' % (message.topic, message.payload))

#Init WebCam
cam = cv2.VideoCapture(0)

MAX_DISCOVERY_RETRIES = 10
GROUP_CA_PATH = "./groupCA/"

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Progressive back off core
backOffCore = ProgressiveBackOffCore()

# Start Discovering GGCs
discoveryInfoProvider = DiscoveryInfoProvider()
discoveryInfoProvider.configureEndpoint(awsiotEndpoint)
discoveryInfoProvider.configureCredentials(rootCAPath, certificatePath, privateKeyPath)
discoveryInfoProvider.configureTimeout(10)  # time out in seconds for discovery request sending/response waiting
retryCount = MAX_DISCOVERY_RETRIES
discovered = False
groupCA = None
coreInfo = None
while retryCount != 0:
    try:
        discoveryInfo = discoveryInfoProvider.discover(thingName=thingName)
        caList = discoveryInfo.getAllCas() #returns GGCore group id and its Root CA cert
        coreList = discoveryInfo.getAllCores() #Returns CoreConnectivityInfo object

        # We only pick the first ca and core info
        groupId, ca = caList[0]
        coreInfo = coreList[0] 
        print("Discovered GGC: %s from Group: %s" % (coreInfo.coreThingArn, groupId))
        
        #Saving Group Root CA cert
        print("Now we persist the connectivity/identity information...")
        groupCA = GROUP_CA_PATH + groupId + "_CA_" + str(uuid.uuid4()) + ".crt"
        if not os.path.exists(GROUP_CA_PATH):
            os.makedirs(GROUP_CA_PATH)
        groupCAFile = open(groupCA, "w")
        groupCAFile.write(ca)
        groupCAFile.close()

        discovered = True
        print("Now proceed to the connecting flow...")
        break
    except DiscoveryInvalidRequestException as e:
        print("Invalid discovery request detected!")
        print("Type: %s" % str(type(e)))
        print("Error message: %s" % e.message)
        print("Stopping...")
        break
    except BaseException as e:
        print("Error in discovery!")
        print("Type: %s" % str(type(e)))
        print("Error message: %s" % e.message)
        retryCount -= 1
        print("\n%d/%d retries left\n" % (retryCount, MAX_DISCOVERY_RETRIES))
        print("Backing off...\n")
        backOffCore.backOff()

if not discovered:
    print("Discovery failed after %d retries. Exiting...\n" % (MAX_DISCOVERY_RETRIES))
    sys.exit(-1)

#After finding greengrass core, time to create MQTT connection with that core.
myAWSIoTMQTTClient = AWSIoTMQTTClient(thingName)
myAWSIoTMQTTClient.configureCredentials(groupCA, privateKeyPath, certificatePath)
myAWSIoTMQTTClient.onMessage = customOnMessage #gets called for every received msg

#one core has many ip address configured in aws console. Thats why for loop
# Iterate through all connection options for the core and use the first successful one
connected = False
for connectivityInfo in coreInfo.connectivityInfoList: 
    currentHost = connectivityInfo.host
    currentPort = connectivityInfo.port
    print("Trying to connect to core at %s:%d" % (currentHost, currentPort))
    myAWSIoTMQTTClient.configureEndpoint(currentHost, currentPort)
    try:
        myAWSIoTMQTTClient.connect()
        connected = True
        break
    except BaseException as e:
        print("Error in connect!")
        print("Type: %s" % str(type(e)))
        print("Error message: %s" % e.message)

if not connected:
    print("Cannot connect to core %s. Exiting..." % coreInfo.coreThingArn)
    sys.exit(-2)

topic = 'handwritingmodel/frames'
myAWSIoTMQTTClient.subscribe(topic, 0, None)
print('Subscribed topic %s:\n' % (topic))
time.sleep(2)

#After finding greengrass core and creating successful MQTT connection, Starting Webcam
while True:        
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.namedWindow('Cam',cv2.WINDOW_NORMAL)
    cv2.imshow("Cam", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        message = {}
        print(f'DEBUG: Size as Numpy array: {frame.nbytes}')
        
        #compressing image and base64 encoding to keep the image size under the AWS MQTT
        #payload limit which is 128KB and send image as json
        img,JPEG = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f'DEBUG: Size as JPEG: {JPEG.nbytes}')

        b64 = b64encode(JPEG)
        print(f'DEBUG: Size as base64: {len(b64)}')
        message['image'] = b64.decode("utf-8")
        messageJson = json.dumps(message)

        #x= len(messageJson.encode("utf-8"))
        myAWSIoTMQTTClient.publish(topic, messageJson, 0)
        #print('Published topic %s: %s\n' % (topic, messageJson))

cam.release()
cv2.destroyAllWindows()

