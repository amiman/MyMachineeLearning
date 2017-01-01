# The idea is to try learirng other ways an object look like from a movie and tracking the detected object across frames.
# Each time we detect an object we track it in the next frames and add the images tp our data set for training the next detector

import glob
import math
import ntpath
import os

import cv2
import dlib
import numpy as np
from skimage import io

from DlibObjectDetection.MyMachineeLearning.CreateXML import DetectionXML


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
    # medianXFlow = np.median(xFlow)
    # medianYFlow = np.median(yFlow)

    medianXFlow = np.mean(xFlow)
    medianYFlow = np.mean(yFlow)

    return medianXFlow,medianYFlow

def TrackFarneback(prevFrame,currentFrame,boundingBox,sizeIncrement):

    # 1 Track it in the current frame
    flow = cv2.calcOpticalFlowFarneback(prev=prevFrame,next=currentFrame, flow=None,pyr_scale=0.5, levels=3,winsize=5,iterations=5,poly_n=7,poly_sigma=1.1,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # 2. Find the flow of the object from the prevoious frame
    medianXFlow, medianYFlow = GetTrackedBox(boundingBox, flow)

    # 3. Add tracked box
    newDetection = dlib.rectangle(int(round((medianXFlow + boundingBox.left())) - sizeIncrement),
                                  int(round((medianYFlow + boundingBox.top()))) - sizeIncrement, int(
            round((medianXFlow + boundingBox.left()))) + boundingBox.width() + sizeIncrement, int(
            round((medianYFlow + boundingBox.top()))) + boundingBox.height() + sizeIncrement)

    return newDetection

def TrackLK(prevFrame,currentFrame,boundingBox,sizeIncrement):

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.3,
                          minDistance=5,
                          blockSize=5)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(10, 10),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001))

    slack = 10
    prevBoundingBox = prevFrame[boundingBox.top()-slack:boundingBox.top()+boundingBox.height()+slack,boundingBox.left()-slack:boundingBox.left()+boundingBox.width()+slack]
    currBoundingBox = currentFrame[boundingBox.top()-slack:boundingBox.top()+boundingBox.height()+slack,boundingBox.left()-slack:boundingBox.left()+boundingBox.width()+slack]

    p0 = cv2.goodFeaturesToTrack(prevBoundingBox, mask=None, **feature_params)
    if(p0 is None):
        return None

    p1, st, err = cv2.calcOpticalFlowPyrLK(prevBoundingBox, currBoundingBox, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]


    # Check if we have good points inside our previous detection
    # pointsForMovmentCalc = []
    # for index, pt in enumerate(good_old):
    #
    #     if pt[0] >= boundingBox.top() and pt[0] <= boundingBox.top()+boundingBox.height():
    #         if pt[1] >= boundingBox.left() and pt[1] <= boundingBox.left() + boundingBox.width():
    #
    #             pointsForMovmentCalc.append((pt,good_new[index]))
    #
    # if(len(pointsForMovmentCalc) > 0):

        # xMovment = np.zeros((len(pointsForMovmentCalc), 1))
        # yMovment = np.zeros((len(pointsForMovmentCalc), 1))
        # index = 0
        #
        # # Use selected points inside the box
        # for pt0,pt1 in pointsForMovmentCalc:
        #
        #     xMovment[index] = (pt1[1] - pt0[1])
        #     yMovment[index] = (pt1[0] - pt0[0])
        #
        #     index += 1

    # else:
    #
    #     return None
    #
    #     # Use general motion estimation
    #     xMovment = np.zeros((len(good_new), 1))
    #     yMovment = np.zeros((len(good_new), 1))
    #     index = 0
    #
    #     # Use selected points inside the box
    #     for index,pt1 in enumerate(good_new):
    #
    #         pt0 = good_old[index]
    #         xMovment[index] = (pt1[0] - pt0[0])
    #         yMovment[index] = (pt1[1] - pt0[1])
    #
    #         index += 1
    #
    # if(len(xMovment) == 0 or len(yMovment) == 0):
    #     return None

    # Use general motion estimation
    xMovment = np.zeros((len(good_new), 1))
    yMovment = np.zeros((len(good_new), 1))
    index = 0

    # Use selected points inside the box
    for index,pt1 in enumerate(good_new):

        pt0 = good_old[index]
        xMovment[index] = (pt1[0] - pt0[0])
        yMovment[index] = (pt1[1] - pt0[1])

        index += 1

    if(len(xMovment) == 0 or len(yMovment) == 0):
        return None

    # Calc movment
    medianXFlow = np.median(xMovment)
    medianYFlow = np.median(yMovment)

    newDetection = dlib.rectangle(int(round((medianXFlow + boundingBox.left()))),
                                  int(round((medianYFlow + boundingBox.top()))), int(
            round((medianXFlow + boundingBox.left()))) + boundingBox.width(), int(
            round((medianYFlow + boundingBox.top()))) + boundingBox.height())

    return newDetection

def TrackCamShift(prevFrame,currentFrame,boundingBox):

    boxCV = (boundingBox.top(),boundingBox.left(),boundingBox.height(),boundingBox.width())
    roi = prevFrame[boxCV[0]:boxCV[0]+boxCV[2],boxCV[1]:boxCV[1]+boxCV[3]]

    # set up the ROI for tracking
    mask = cv2.inRange(roi, np.array((0.)), np.array((255.)))
    roi_hist = cv2.calcHist([roi], [0], mask, [255], [0, 255])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # Convert current image to hsv
    dst = cv2.calcBackProject([currentFrame.astype(np.float32)], [0], roi_hist, [0,255], 1)

    # apply CamShift to get the new location
    ret, track_window = cv2.CamShift(dst, boxCV, term_crit)

    # Get new detection rectangle
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)

    newDetection =  dlib.rectangle(int(pts[0][0]),int(pts[1][1]),int(pts[1][0]),int(pts[3][1]))

    # Return new location
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

detectorPath = os.path.normpath("C:\Users\HiPark\PycharmProjects\HiPark\DlibObjectDetection\MyMachineeLearning\detector.svm")
inputDirectory = os.path.normpath("C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In\\")
outputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\IN-AN_7822_20161201_115848\DlibObjectDetection\out\\")
xmlOutput = outputPath + "\\detecions.xml"

# This is to compensate the size of the rectangle needs to be because of the direction of the car
# , i.e the car is going stright thigs get closer to camera are bigger therfore the size of the boundnig box needs to be bigger
# This of course needs to be a function of the direction of the car speed and the FPS. Another way to get a correct measurment could be
# using some registration techniqe that calculates also scale change.
sizeIncrement = 0

# Intialize detector
detector = dlib.simple_object_detector(detectorPath)

# Create an xml detection file
xmlDetection = DetectionXML(xmlOutput)

startFrame = 785
endFrame = 1250
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
    # print("Processing file: {}".format(f))
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

            # newDetection = TrackFarneback(prevImage,cvCurrent,boundingBoxPrev,sizeIncrement)
            newDetection = TrackLK(prevImage, cvCurrent, boundingBoxPrev, sizeIncrement)
            # newDetection = TrackCamShift(prevImage, cvCurrent, boundingBoxPrev)

            if(newDetection is None):
                continue

            # 3.1.3 Check box
            # print str(newDetection.top()) + " " + str(newDetection.left()) + " " + str(newDetection.right()) + " " + str(newDetection.bottom())
            if(newDetection.top() < 0 or newDetection.left() < 0 or newDetection.right() >= cols-10 or newDetection.bottom() >= rows-10):
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
                if(distanceB(centerBoxCurrentX,centerBoxCurrentY,centerBoxPrevX,centerBoxPrevY) < 100):
                    foundMatch = True
                    break

            if(not foundMatch):

                # Find new detection
                # newDetection = TrackFarneback(prevImage, cvCurrent, boundingBoxPrev, sizeIncrement)
                newDetection = TrackLK(prevImage, cvCurrent, boundingBoxPrev, sizeIncrement)
                # newDetection = TrackCamShift(prevImage, cvCurrent, boundingBoxPrev)

                if (newDetection is None):
                    continue

                # 4.2.3 Check box
                if (newDetection.top() < 0 or newDetection.left() < 0 or newDetection.right() >= cols-10 or newDetection.bottom() >= rows-10):
                    continue

                # 4.2.4 Save the image
                # SaveImage(f, outputPath, trackingIndex,
                #  cvCurrent[newDetection.top():newDetection.top() + newDetection.height(),
                #           newDetection.left():newDetection.left() + newDetection.width()])
                trackingIndex += 1

                # 4.2.5 Add this new tracked image to the detections
                newDetections.append(newDetection)

    # Add new detections to current detections
    for a, boundingBox in enumerate(newDetections):
        print boundingBox
        detections.append(boundingBox)

    # 5. After we finish learning from tracking add the detections to xml detection
    if(len(detections) > 0):
        xmlDetection.addImage(f, detections)

    # Update variales for next detection
    prevImage = cvCurrent
    prevDetctions = detections


# Save xml detections file
xmlDetection.exportXML()