import os
import sys

import DlibObjectDetection.MyMachineeLearning.Code.ObjectTrainner
import DlibObjectDetection.MyMachineeLearning.Code.ObjectDetector
import DlibObjectDetection.MyMachineeLearning.Code.LearningTracker
import DlibObjectDetection.MyMachineeLearning.Code.DetectionXML

import DlibObjectDetection.MyMachineeLearning.Code.Util

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

# InputImageDirectory = os.path.normpath("C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In")
InputImageDirectory = os.path.normpath("C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In")
OutputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\Output")

ObjectSVMOutput = OutputPath + "\detector.svm"
DetectionOutput = InputImageDirectory + "\detections2.xml"
DetectionOutputTemp = InputImageDirectory + "\detections2FrontAndBack.xml"

DetectionTag = "box"

# TrainFolder = InputImageDirectory
# TrainXMLFile = "detections2.xml"

# Flags
Verbose = True
Down_Sample = False
Crop_For_RAM_Saving = False

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
    DlibObjectDetection.MyMachineeLearning.Code.Util.DownSampleTrainingFolder(TrainFolder, TrainFolderDownSample, 3, ".jpg", TrainFolder + "\\" + TrainXMLFile, TrainDownsampledXML)

    TrainFolder = TrainFolderDownSample

    if (Verbose):
        print("DownSampling input folder")

    InputImageDirectoryDownSample = InputImageDirectory + "\DownSample"
    InputImageDirectoryDownSample = TrainFolder + "\DownSample"
    try:
        os.mkdir(InputImageDirectoryDownSample)
    except:
        print("Folder allready exsits")

    DlibObjectDetection.MyMachineeLearning.Code.Util.DownSampleFolder(InputImageDirectory, InputImageDirectoryDownSample, 3, ".bmp")
    InputImageDirectory = InputImageDirectoryDownSample
    DetectionOutput = InputImageDirectory + "\detectionsDownSampled.xml"
    DetectionOutputTemp = InputImageDirectory + "\detections2.xml"

# Start loop on stages 1- 4
numbreOfOldDetection = 0
numberOfNewDetection = 1
while(numberOfNewDetection > numbreOfOldDetection):

################################################## 0. Preprocessiong #############################################################
    if(Crop_For_RAM_Saving):

        TrainFolderCrop = TrainFolder + "\Crop"
        TrainCropdXML = TrainFolderCrop + "\\" + TrainXMLFile
        try:
            os.mkdir(TrainFolderCrop)
        except:
            print("Folder allready exsits")

        DlibObjectDetection.MyMachineeLearning.Code.Util.CropImageAccodringToDetections(TrainFolderCrop, TrainFolder + "\\" + TrainXMLFile, TrainCropdXML)
        TrainFolder = TrainFolderCrop

################################################# 1. Learn simpel linear classsfier #############################################################

    if (Verbose):
        print("Training object detector")

    trainer = DlibObjectDetection.MyMachineeLearning.Code.ObjectTrainner.ObjectTrainner(TrainFolder, TestFolder, TrainXMLFile)
    trainer.RunTraining(ObjectSVMOutput)

################################################## 2. Detect objects in videos #############################################################

    if (Verbose):
        print("Detecting object in movie")

    detector = DlibObjectDetection.MyMachineeLearning.Code.ObjectDetector.ObjectDetector(ObjectSVMOutput)
    detector.Detect(InputImageDirectory,ImageFileExtension,DetectionOutput,verbose=True)

################################################## 3. Track objects in videos #############################################################

    if (Verbose):
        print("Tracking detections")

    learnTracker = DlibObjectDetection.MyMachineeLearning.Code.LearningTracker.LearningTracker(InputImageDirectory, ImageFileExtension, DetectionOutput, DetectionOutput)

    # Learn using tracking tracking the image forward
    learnTracker.Learn(1)

    # Save xml detections file
    learnTracker.ExportXML()

    learnTracker = DlibObjectDetection.MyMachineeLearning.Code.LearningTracker.LearningTracker(InputImageDirectory, ImageFileExtension, DetectionOutput, DetectionOutput)

    # Learn using tracking tracking the image backward
    learnTracker.Learn(0)

    # Save xml detections file
    learnTracker.ExportXML()

################################################## 4. Update for next iteration #############################################################

    if (Verbose):
        print("Update data for next itreation")

    # DetectionOutput = DetectionOutputTemp
    numbreOfOldDetection = numberOfNewDetection

    TrainFolder = InputImageDirectory

    # Count the number of new detections
    xmlRead = DlibObjectDetection.MyMachineeLearning.Code.DetectionXML.DetectionXML(DetectionOutput, 1)

    numberOfNewDetection = xmlRead.getNubmerOfTagInFile(DetectionTag)

    TrainXMLFile = "detections2.xml"

    # Crop_For_RAM_Saving = True

    print numbreOfOldDetection
    print numberOfNewDetection

