# The idea is to try learirng other ways an object look like from a movie and tracking the detected object across frames.
# Each time we detect an object we track it in the next frames and add the images tp our data set for training the next detector

import cv2
import numpy as np
import sys
import glob
import os
import dlib
from skimage import io
import math
from matplotlib import pyplot as plt
import ntpath

from CreateXML import DetectionXML

def getBoxCenter(box):
    centerBoxCurrentY = (box.top() + box.top() + box.height()) / 2
    centerBoxCurrentX = (box.left() + box.left() + box.width()) / 2

    return centerBoxCurrentX ,centerBoxCurrentY

def distanceB(a,b,c,d):

    return math.sqrt((a-c)*(a-c)+(b-d)*(b-d))

def GetTrackedBox(box,flow):

    # nextPixels = []
    xFlow = np.zeros((box.height() * box.width(),1))
    yFlow = np.zeros((box.height() * box.width(), 1))

    index = 0
    xBorder = min(box.left() + box.width(),flow.shape[0])
    yBorder = min(box.top()+box.height(),flow.shape[1])

    for i in range(box.top(),yBorder,1):
        for j in range(box.left(), xBorder,1):
            yFlow[index] = flow[i, j][1]
            xFlow[index] = flow[i, j][0]
            index += 1

    # Calculate the median shift of the tracked box
    medianXFlow = np.median(xFlow)
    medianYFlow = np.median(yFlow)

    return medianXFlow,medianYFlow

def Track(prevFrame,currentFrame,boundingBox,sizeIncrement):

    # 1 Track it in the current frame
    flow = cv2.calcOpticalFlowFarneback(prevFrame,currentFrame, None, 0.5, 3, 50, 5, 7, 1.5, 1)

    # 2. Find the flow of the object from the prevoious frame
    medianXFlow, medianYFlow = GetTrackedBox(boundingBox, flow)

    # 3. Add tracked box
    newDetection = dlib.rectangle(int(round((medianXFlow + boundingBox.left())) - sizeIncrement),
                                  int(round((medianYFlow + boundingBox.top()))) - sizeIncrement, int(
            round((medianXFlow + boundingBox.left()))) + boundingBox.width() + sizeIncrement, int(
            round((medianYFlow + boundingBox.top()))) + boundingBox.height() + sizeIncrement)

    return newDetection

def SaveImage(filePath,outputDir,index,img):

    # Crate file name
    fileName = ntpath.basename(filePath)
    fileName = os.path.splitext(fileName)[0]
    newDetectionPath = outputDir + "\\" + fileName + "_" + str(index) + ".bmp"

    # Save
    cv2.imwrite(newDetectionPath, img)

# Params
# detectorPath = sys.argv[1]
# inputDirectory = sys.argv[2]

detectorPath = os.path.normpath("C:\Users\HiPark\PycharmProjects\HiPark\DlibObjectDetection\detector.svm")
inputDirectory = os.path.normpath("C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In\\")
outputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\IN-AN_7822_20161201_115848\DlibObjectDetection\out\\")
xmlOutput = outputPath + "\\detecions.xml"

# This is to compensate the size of the rectangle needs to be because of the direction of the car
# , i.e the car is going stright thigs get closer to camera are bigger therfore the size of the boundnig box needs to be bigger
# This of course needs to be a function of the direction of the car speed and the FPS. Another way to get a correct measurment could be
# using some registration techniqe that calculates also scale change.
sizeIncrement = 1

# Intialize detector
detector = dlib.simple_object_detector(detectorPath)

# Create an xml detection file
xmlDetection = DetectionXML(xmlOutput)

startFrame = 0
endFrame = 2000
currentFrame = 0

# Go over the movie and track detected object
prevImage = np.zeros((1,1))
for f in glob.glob(os.path.join(inputDirectory, "*.bmp")):

    if(startFrame > currentFrame):
        currentFrame += 1
        continue
    elif currentFrame > endFrame:
        break
    currentFrame += 1

    # 1. Read image
    print("Processing file: {}".format(f))
    img = io.imread(f)
    cvCurrent = cv2.imread(f,0)

    fileName = f.endswith("." + "PNG")

    # 2. Run object detection
    detections = detector(img)

    cols = cvCurrent.shape[1]
    rows = cvCurrent.shape[0]

    if (len(detections) == 0):

        # Chcek if we had detections last frame
        if(prevImage.shape[0] == 1):
            prevDetctions = []
            prevImage = np.zeros((1, 1))
            continue

    # 3. We have detections from previous frame but not from current frame try to track them
        newDetections = dlib.rectangles()
        trackingIndex = 1
        for detectedPrev, boundingBoxPrev in enumerate(prevDetctions):

            newDetection = Track(prevImage,cvCurrent,boundingBoxPrev,sizeIncrement)

            # 3.1.3 Check box
            # print str(newDetection.top()) + " " + str(newDetection.left()) + " " + str(newDetection.right()) + " " + str(newDetection.bottom())
            if(newDetection.top() < 0 or newDetection.left() < 0 or newDetection.right() >= cols or newDetection.bottom() >= rows):
                continue

            # 3.1.4 Save the image
            # SaveImage(f,outputPath,trackingIndex,cvCurrent[newDetection.top():newDetection.top() + newDetection.height(),newDetection.left():newDetection.left() + newDetection.width()])
            trackingIndex += 1

            # 3.1.4 Add this new tracked image to the detections
            newDetections.append(newDetection)

    # 4. We have detected object in this frame and also in the previous one

    else:
        # 4.1 Chcek if we detected objects in the last frame if yes find correlated detections
        if(prevImage.shape[0] == 1):
            prevImage = cvCurrent
            prevDetctions = detections
            continue

        # 4.1.1 Go over last detections and find corrleations with current detections
        newDetections = dlib.rectangles()
        for detectedPrev, boundingBoxPrev in enumerate(prevDetctions):
            foundMatch = False
            trackingIndex = 1
            for detectedCurrent, boundingBoxCurrent in enumerate(detections):

                # 4.1.2 Find center of boxes
                centerBoxCurrentX,centerBoxCurrentY = getBoxCenter(boundingBoxCurrent)
                centerBoxPrevX, centerBoxPrevY = getBoxCenter(boundingBoxPrev)

                # 4.1.3 Find the distances between boxes
                if(distanceB(centerBoxCurrentX,centerBoxCurrentY,centerBoxPrevX,centerBoxPrevY) < 50):
                    foundMatch = True
                    break

            if(not foundMatch):

                # Find new detection
                newDetection = Track(prevImage, cvCurrent, boundingBoxPrev, sizeIncrement)

                # 4.2.3 Check box
                if (newDetection.top() < 0 or newDetection.left() < 0 or newDetection.right() >= cols or newDetection.bottom() >= rows):
                    continue

                # 4.2.4 Save the image
                # SaveImage(f, outputPath, trackingIndex,
                #  cvCurrent[newDetection.top():newDetection.top() + newDetection.height(),
                #           newDetection.left():newDetection.left() + newDetection.width()])
                trackingIndex += 1

                # 4.2.5 Add this new tracked image to the detections
                newDetections.append(newDetection)

    # 4.2 After we finish learing from tracking add the detections to xml detection
    if(len(detections) > 0):
        xmlDetection.addImage(f, newDetections)

    # Add new detections to current detections
    for a, boundingBox in enumerate(newDetections):
        detections.append(boundingBox)

    # Update variales for next detection
    prevImage = cvCurrent
    prevDetctions = detections


# Save xml detections file
xmlDetection.exportXML()