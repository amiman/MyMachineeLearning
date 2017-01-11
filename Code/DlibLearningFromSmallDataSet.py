import os
import sys

import ObjectTrainner
import ObjectDetector
import LearningTracker
import DetectionXML

import Util

# The idea of learning from a small data set is the following:
# 1. Use a simple linear classifier that can be trained using small data set and give reasonable results
# 2. Use the trained classifier to detect the object in a video
# 3. Start tracking the detected object across frames and add the new learned examples to a new data set
# 4. Retrain the classifier again with the larger data set and do stages 1-3.
# 5. After exhausting stages 1-3 do data augmentation involving light,noise and more manipulations
# 6. When there is enough data move fom simple linear classifier to training using deep learning


# Params
TrainFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\TrainingImages")
TestFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\TestingImages")
TrainXMLFile = "training.xml"
ImageFileExtension = ".bmp"

InputImageDirectory = os.path.normpath("C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In")
OutputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\Output")

ObjectSVMOutput = OutputPath + "\detector.svm"
DetectionOutput = InputImageDirectory + "\detectionsDownSampled.xml"
DetectionOutputTemp = InputImageDirectory + "\detections2.xml"

DetectionTag = "box"

# TrainFolder = InputImageDirectory
# TrainXMLFile = "detections2.xml"

# Flags
Verbose = True
Down_Sample = True

# Preprocessing - Do several oprations before running the main algorithm
if(Down_Sample):

    if(Verbose):
        print("DownSampling training folder")

    # Crate a folder for the downsampled images
    TrainFolderDownSample = TrainFolder + "\DownSample"
    TrainDownsampledXML = TrainFolderDownSample + "\\" + TrainXMLFile
    try:
        os.mkdir(TrainFolderDownSample)
    except:
        print("Folder allready exsits")

    # Downsample images
    Util.DownSampleTrainingFolder(TrainFolder,TrainFolderDownSample,2,".jpg",TrainFolder + "\\" + TrainXMLFile, TrainDownsampledXML )

    TrainFolder = TrainFolderDownSample

    if (Verbose):
        print("DownSampling input folder")

    InputImageDirectoryDownSample = TrainFolder + "\DownSample"
    try:
        os.mkdir(InputImageDirectoryDownSample)
    except:
        print("Folder allready exsits")

    Util.DownSampleFolder(InputImageDirectory, InputImageDirectoryDownSample, 2, ".bmp")
    InputImageDirectory = InputImageDirectoryDownSample

# Start loop on stages 1- 4
# numbreOfOldDetection = 0
# numberOfNewDetection = 1
# while(numberOfNewDetection > numbreOfOldDetection):

################################################## 1. Learn simpel linear classsfier #############################################################

    if (Verbose):
        print("Training object detector")

    trainer = ObjectTrainner.ObjectTrainner(TrainFolder,TestFolder,TrainXMLFile)
    trainer.RunTraining(ObjectSVMOutput)

################################################## 2. Detect objects in videos #############################################################

    if (Verbose):
        print("Detecting object in movie")

    detector = ObjectDetector.ObjectDetector(ObjectSVMOutput)
    detector.Detect(InputImageDirectory,ImageFileExtension,DetectionOutput,verbose=True)

################################################## 3. Track objects in videos #############################################################
#
    if (Verbose):
        print("Tracking detections")

#     learnTracker = LearningTracker.LearningTracker(InputImageDirectory, ImageFileExtension, DetectionOutput, DetectionOutputTemp)
#
#     # Learn using tracking tracking the image forward
#     learnTracker.Learn(1)
#
#     # Save xml detections file
#     learnTracker.ExportXML()
#
#     learnTracker = LearningTracker.LearningTracker(InputImageDirectory, ImageFileExtension, DetectionOutputTemp, DetectionOutputTemp)
#
#     # Learn using tracking tracking the image backward
#     learnTracker.Learn(0)
#
#     # Save xml detections file
#     learnTracker.ExportXML()
#
# ################################################## 4. Update for next iteration #############################################################
#
    if (Verbose):
        print("Update data for next itreation")

#     DetectionOutput = DetectionOutputTemp
#     numbreOfOldDetection = numberOfNewDetection
#
#     TrainFolder = InputImageDirectory
#
#
#     # Count the number of new detections
#     xmlRead = DetectionXML.DetectionXML(DetectionOutput,1)
#
#     numberOfNewDetection = xmlRead.getNubmerOfTagInFile(DetectionTag)

