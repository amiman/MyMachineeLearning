# The idea is to try learirng other ways an object look like from a movie and tracking the detected object across frames.
# Each time we detect an object we track it in the next frames and add the images tp our data set for training the next detector

import ntpath
import os

import cv2

from DlibObjectDetection.MyMachineeLearning.Code.LearningTracker import LearningTracker

def SaveImage(filePath,outputDir,index,img):

    # Crate file name
    fileName = ntpath.basename(filePath)
    fileName = os.path.splitext(fileName)[0]
    newDetectionPath = outputDir + "\\" + fileName + "_" + str(index) + ".bmp"

    # Save
    cv2.imwrite(newDetectionPath, img)

# Params
inputDirectory = os.path.normpath("C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In\\")
outputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\IN-AN_7822_20161201_115848\DlibObjectDetection\out\\")

xmlInput = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\IN-AN_7822_20161201_115848\DlibObjectDetection\detection.xml")
xmlOutput = outputPath + "\\detecions.xml"

# Crate learning tracker
learnTracker = LearningTracker(inputDirectory, ".bmp", xmlInput, xmlOutput)

# Learn using tracking
learnTracker.Learn(1)

# Save xml detections file
learnTracker.ExportXML()

learnTracker = LearningTracker(inputDirectory, ".bmp", xmlOutput, xmlOutput)

# Learn using tracking
learnTracker.Learn(0)

# Save xml detections file
learnTracker.ExportXML()
