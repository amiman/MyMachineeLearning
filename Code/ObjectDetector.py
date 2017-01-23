import glob
import os

import dlib
from skimage import io

import DetectionXML

class ObjectDetector(object):

    def __init__(self, pathSVMClassfier):

        self.detector = dlib.simple_object_detector(pathSVMClassfier)

    def Detect(self, inputDir, imageExtension, outputXML, verbose=False):

        # Create an xml detection file
        self.xmlDetection = DetectionXML.DetectionXML(outputXML, 0)

        # Run the detecro on the movie
        imageExtension = "*" + imageExtension
        for f in glob.glob(os.path.join(inputDir, imageExtension)):

            if(verbose):
                print("Processing file: {}".format(f))

            # Read image
            img = io.imread(f)

            # Detect object in image
            dets = self.detector(img)

            if (verbose):
                print("Number of object detected: {}".format(len(dets)))
                for k, d in enumerate(dets):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        k, d.left(), d.top(), d.right(), d.bottom()))

            # Save detection
            if (len(dets) > 0):
                self.xmlDetection.addImage(f, dets)


    def ExportXMLFile(self):

        # Export file
        self.xmlDetection.exportXML()

    def GetXMLFile(self):

        return self.xmlDetection
