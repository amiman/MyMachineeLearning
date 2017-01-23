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

class Algorithm(object):

    def __init__(self, Verbose, TrainFolder, TestFolder, TrainXML, OutputPath, InputImageDirectory,ImageFileExtension):

        self.Verbose = Verbose
        self.TrainFolder = TrainFolder
        self.TestFolder = TestFolder
        self.TrainXMLFile = TrainXML
        self.OutputPath = OutputPath

        self.InputImageDirectory = InputImageDirectory
        self.ImageFileExtension = ImageFileExtension

        self.DetectionXMLName = "training.xml"
        self.ObjectSVMOutput = self.OutputPath + "\detector.svm"
        self.DetectionOutput = InputImageDirectory + "\\" + self.DetectionXMLName
        self.DetectionOutputTemp = InputImageDirectory + "\detectionsFinalTemp.xml"

        self.DetectionTag = "box"

    def Run(self):

        # Start loop on stages 1- 4
        numbreOfOldDetection = 0
        numberOfNewDetection = 1
        FirstIter = True
        while (numberOfNewDetection > numbreOfOldDetection):

            ################################################## 0. Preprocessiong #############################################################

            ################################################# 1. Learn simpel linear classsfier #############################################################

            if (self.Verbose):
                print("Training object detector")

            trainer = ObjectTrainner.ObjectTrainner(self.TrainFolder, self.TestFolder, self.TrainXMLFile)
            trainer.RunTraining(self.ObjectSVMOutput)

            ################################################## 2. Detect objects in videos #############################################################

            if (self.Verbose):
                print("Detecting object in movie")

            detector = ObjectDetector.ObjectDetector(self.ObjectSVMOutput)
            detector.Detect(self.InputImageDirectory, self.ImageFileExtension, self.DetectionOutput, verbose=True)
            xmlDetection = detector.GetXMLFile()

            if(FirstIter):
                xmlOld = DetectionXML.DetectionXML(os.path.join(self.TrainFolder, self.TrainXMLFile),0)
                xmlOldDetections = xmlOld.getImageDetections()
                xmlDetection.addImageFrameDetections(xmlOldDetections)
                FirstIter = False

            xmlDetection.exportXML()
            ################################################## 3. Track objects in videos #############################################################

            if (self.Verbose):
                print("Tracking detections")

            learnTracker = LearningTracker.LearningTracker(self.InputImageDirectory, self.ImageFileExtension, self.DetectionOutput,
                                                           self.DetectionOutput)

            # Learn using tracking tracking the image forward
            learnTracker.Learn(1)

            # Save xml detections file
            learnTracker.ExportXML()

            # learnTracker = LearningTracker.LearningTracker(self.InputImageDirectory, self.ImageFileExtension, self.DetectionOutput,
            #                                                self.DetectionOutput)
            #
            # # Learn using tracking tracking the image backward
            # learnTracker.Learn(0)
            #
            # # Save xml detections file
            # learnTracker.ExportXML()

            ################################################## 4. Update for next iteration #############################################################

            if (self.Verbose):
                print("Update data for next itreation")

            numbreOfOldDetection = numberOfNewDetection

            self.TrainFolder = self.InputImageDirectory

            # Count the number of new detections
            xmlRead = DetectionXML.DetectionXML(self.DetectionOutput, 1)

            numberOfNewDetection = xmlRead.getNubmerOfTagInFile(self.DetectionTag)

            self.TrainXMLFile = self.DetectionXMLName

            # Crop_For_RAM_Saving = True

            print numbreOfOldDetection
            print numberOfNewDetection
