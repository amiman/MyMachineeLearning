import ntpath
import math

import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt

import DetectionXML
import ImageDirectory

class LearningTracker(object):

    def __init__(self, inputImageDir, imageFileType, detectionXMLFile, outputDetectionXMLFile):

        self.detectionFilePath = detectionXMLFile

        self.detectionXML = DetectionXML.DetectionXML(detectionXMLFile,1)

        self.outputDetectionXML = DetectionXML.DetectionXML(outputDetectionXMLFile,0)

        self.inputImageDir = ImageDirectory.ImageDirectory(inputImageDir, imageFileType)

        ## Luckas candea tracking params

        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=1000,
                              qualityLevel=0.3,
                              minDistance=5,
                              blockSize=5)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(10, 10),
                         maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001))

        self.slack = 10

    def GetTrackedBox(self, box, flow):

        # nextPixels = []
        xFlow = np.zeros((box.height() * box.width(), 1))
        yFlow = np.zeros((box.height() * box.width(), 1))

        index = 0
        xBorder = min(box.left() + box.width(), flow.shape[0])
        yBorder = min(box.top() + box.height(), flow.shape[1])

        for i in range(box.top(), yBorder, 1):
            for j in range(box.left(), xBorder, 1):
                yFlow[index] = flow[i, j][1]
                xFlow[index] = flow[i, j][0]
                index += 1

        # Calculate the median shift of the tracked box
        # medianXFlow = np.median(xFlow)
        # medianYFlow = np.median(yFlow)

        meanXFlow = np.mean(xFlow)
        meanYFlow = np.mean(yFlow)

        return meanXFlow, meanYFlow

    def TrackFarneback(self, prevFrame,currentFrame,boundingBox,sizeIncrement):

        # 1 Track it in the current frame
        flow = cv2.calcOpticalFlowFarneback(prev=prevFrame, next=currentFrame, flow=None, pyr_scale=0.5, levels=3,
                                            winsize=5, iterations=5, poly_n=7, poly_sigma=1.1,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # 2. Find the flow of the object from the prevoious frame
        medianXFlow, medianYFlow = self.GetTrackedBox(boundingBox, flow)

        # 3. Add tracked box
        newDetection = dlib.rectangle(int(round((medianXFlow + boundingBox.left())) - sizeIncrement),
                                      int(round((medianYFlow + boundingBox.top()))) - sizeIncrement, int(
                round((medianXFlow + boundingBox.left()))) + boundingBox.width() + sizeIncrement, int(
                round((medianYFlow + boundingBox.top()))) + boundingBox.height() + sizeIncrement)

        return newDetection

    def TrackLK(self, prevFrame, currentFrame, boundingBox):

        # Define the tracked window using the prevoius detection
        prevBoundingBox = prevFrame[boundingBox.top() - self.slack:boundingBox.top() + boundingBox.height() + self.slack,
                          boundingBox.left() - self.slack:boundingBox.left() + boundingBox.width() + self.slack]
        currBoundingBox = currentFrame[boundingBox.top() - self.slack:boundingBox.top() + boundingBox.height() + self.slack,
                          boundingBox.left() - self.slack:boundingBox.left() + boundingBox.width() + self.slack]

        # Find points for trackfing in the prevouis detection
        p0 = cv2.goodFeaturesToTrack(prevBoundingBox, mask=None, **self.feature_params)

        # If we didn't found any points for tracking return
        if (p0 is None):
            return None

        # Run luckas cande optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prevBoundingBox, currBoundingBox, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Use general motion estimation
        xMovment = np.zeros((len(good_new), 1))
        yMovment = np.zeros((len(good_new), 1))
        index = 0

        # Use selected points inside the box
        for index, pt1 in enumerate(good_new):
            pt0 = good_old[index]
            xMovment[index] = (pt1[0] - pt0[0])
            yMovment[index] = (pt1[1] - pt0[1])

            index += 1

        if (len(xMovment) == 0 or len(yMovment) == 0):
            return None

        # Calc movment
        medianXFlow = np.median(xMovment)
        medianYFlow = np.median(yMovment)

        newDetection = dlib.rectangle(int(round((medianXFlow + boundingBox.left()))),
                                      int(round((medianYFlow + boundingBox.top()))), int(
                round((medianXFlow + boundingBox.left()))) + boundingBox.width(), int(
                round((medianYFlow + boundingBox.top()))) + boundingBox.height())

        return newDetection

    def Learn(self):

        # Get the detections from the xml file
        FrameDataList = self.detectionXML.getImageDetections()

        # Go over the detections and start track the detections
        index = 0
        for frame in FrameDataList:

            if index == 0:
                # self.outputDetectionXML.addImage(frame.GetFilePath(),frame.GetDetections())
                prevFrame = frame
                index += 1
                continue

            currnetFrame = frame

            # Check if the current frame and the prevframe are one after the other
            print prevFrame.GetFrameBaseName()

            prevFrameNumber = self.inputImageDir.GetFileIndexByName(prevFrame.GetFrameBaseName())
            currentFrameNumber = self.inputImageDir.GetFileIndexByName(currnetFrame.GetFrameBaseName())

            if currentFrameNumber == prevFrameNumber + 1:

                # We have two successive frames with detections find which detections correpononds to the same object wnd which are not.
                # Track the ones that don't have successive match
                self.TrackingTwoSuccessiveFrames(prevFrame,currnetFrame)

            else:

                # The prevous frame who add detections and the current one are not successive start tracking from the current image to the next detection
                self.TrackingFrame(prevFrame, prevFrameNumber, currentFrameNumber )


            # Update data for next iteration
            prevFrame = currnetFrame
            index += 1

    def TrackingTwoSuccessiveFrames(self, prevFrame,currentFrame):

        # 1. Go over last detections and find corrleations with current detections
        newDetections = dlib.rectangles()
        detections = currentFrame.GetDetections()
        for detectedPrev, boundingBoxPrev in enumerate(prevFrame.GetDetections()):
            foundMatch = False
            trackingIndex = 1

            prevImage = cv2.imread(prevFrame.GetFilePath(),0)

            for detectedCurrent, boundingBoxCurrent in enumerate(detections):

                # 1.2 Find center of boxes
                centerBoxCurrentX,centerBoxCurrentY = self.GetBoxCenter(boundingBoxCurrent)
                centerBoxPrevX, centerBoxPrevY = self.GetBoxCenter(boundingBoxPrev)

                # 1.3 Find the distances between boxes
                if(self.DistanceB(centerBoxCurrentX,centerBoxCurrentY,centerBoxPrevX,centerBoxPrevY) < 100):
                    foundMatch = True
                    break

            if(not foundMatch):

                # 2.1 Find new detection
                currentImage = cv2.imread(currentFrame.GetFilePath(),0)
                newDetection = self.TrackLK(prevImage, currentImage, boundingBoxPrev)

                if (newDetection is None):
                    continue

                # 2.2 Check box
                if (newDetection.top() < 0 or newDetection.left() < 0 or newDetection.right() >= currentImage.shape[1]-10 or newDetection.bottom() >= currentImage.shape[0]-10):
                    continue

                print newDetection
                trackingIndex += 1

                # 2.3 Add this new tracked image to the detections
                newDetections.append(newDetection)

        # Add the new detections
        for a, boundingBox in enumerate(newDetections):
            detections.append(boundingBox)

        self.outputDetectionXML.addImage(currentFrame.GetFilePath(), detections)

    def TrackingFrame(self, prevFrame, currentFrameIndex, nextFrameIndex):

        # Start tracking from current image unitlo we get to the next image index or we can't track anymore
        files = self.inputImageDir.GetFilesPathUsingIndexes(currentFrameIndex, nextFrameIndex)
        index = 0
        for file in files:

            # The first index is the current frame
            if index == 0:
                prevDetections = prevFrame.GetDetections()
                self.outputDetectionXML.addImage(prevFrame.GetFilePath(),prevDetections)
                prevImage = cv2.imread(prevFrame.GetFilePath(),0)
                index += 1
                continue

            currentImage = cv2.imread(file,0)
            newDetections = dlib.rectangles()

            for detectedPrev, boundingBoxPrev in enumerate(prevDetections):

                # Track current detections
                newDetection = self.TrackLK(prevImage, currentImage, boundingBoxPrev)

                if (newDetection is None):
                    continue

                print newDetection

                # Check box
                if (newDetection.top() < 0 or newDetection.left() < 0 or newDetection.right() >= prevImage.shape[1] - 10 or newDetection.bottom() >= prevImage.shape[0] - 10):
                    continue

                # 3.1.4 Add this new tracked image to the detections
                newDetections.append(newDetection)

            if(len(newDetections) > 0):
                self.outputDetectionXML.addImage(file,newDetections)

            prevImage = currentImage
            prevDetections = newDetections
            index += 1

    def GetBoxCenter(self, box):
        centerBoxCurrentY = (box.top() + box.top() + box.height()) / 2
        centerBoxCurrentX = (box.left() + box.left() + box.width()) / 2

        return centerBoxCurrentX ,centerBoxCurrentY

    def DistanceB(self, a,b,c,d):

        return math.sqrt((a-c)*(a-c)+(b-d)*(b-d))

    def ExportXML(self):

        self.outputDetectionXML.exportXML()

