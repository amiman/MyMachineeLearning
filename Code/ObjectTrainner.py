import os

import dlib


class ObjectTrainner(object):

    def __init__(self, trainFolder, testFolder, xmlTrainFileName = "training.xml", xmlTestFileName = "testing.xml"):

        self.trainFolder = trainFolder
        self.testFolder = testFolder

        self.options = dlib.simple_object_detector_training_options()
        self.options.add_left_right_image_flips = True

        self.options.C = 5

        self.options.num_threads = 3
        self.options.be_verbose = True

        self.trainingXmlPath = os.path.join(self.trainFolder, xmlTrainFileName)
        self.testingXmlPath = os.path.join(self.testFolder, xmlTestFileName)

    def RunTraining(self, outputDetectorName="detector.svm"):
        self.detectorName = outputDetectorName
        dlib.train_simple_object_detector(self.trainingXmlPath, outputDetectorName, self.options)


    def RunTesting(self):

        print("Training accuracy: {}".format(
            dlib.test_simple_object_detector(self.trainingXmlPath, self.detectorName )))

        print("Testing accuracy: {}".format(
            dlib.test_simple_object_detector(self.testingXmlPath, self.detectorName )))

