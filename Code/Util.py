import glob
import os
import ntpath

import cv2
import dlib

import DetectionXML

def DownSampleTrainingFolder(folderIn, folderOut, factor, imageExtension, inputXmlPath, outputXmlPath):

    imageExtension = "*" + imageExtension

    # Go over the files and dowensample them
    for f in glob.glob(os.path.join(folderIn, imageExtension)):

        # Read image
        Iin = cv2.imread(f,0)
        row = Iin.shape[0]
        col = Iin.shape[1]

        # Downsample image
        newSize = (int(round(col / factor)), int(round(row / factor)))
        Iout = cv2.resize(Iin,newSize)

        # Save image
        fileName = ntpath.basename(f)
        imageFilePath = folderOut + "\\" + fileName
        print imageFilePath
        cv2.imwrite(imageFilePath,Iout)

    # Now Go over the xml file and downsample the detections
    xmlIn = DetectionXML.DetectionXML(inputXmlPath, 1)
    xmlOut = DetectionXML.DetectionXML(outputXmlPath, 0)

    # Get the detection boxes
    frameData = xmlIn.getImageDetections()

    # Go over each detection and change the size by factor
    for frame in frameData:

        downSampledDetections = dlib.rectangles()
        for box in frame.GetDetections():

            newBox = dlib.rectangle(int(round((box.left()/factor))),
                           int(round((box.top()/factor))), int(round((box.left() / factor))) + int(round((box.width()/factor))), int(round((box.top()/factor))) + int(round((box.height()/factor))))

            downSampledDetections.append(newBox)

        # Add downsampled detections to a new file
        xmlOut.addImage(frame.GetFilePath(),downSampledDetections)

    # Save the new downsampled xml file
    xmlOut.exportXML()

def DownSampleFolder(folderIn, folderOut, factor, imageExtension):

    imageExtension = "*" + imageExtension

    # Go over the files and dowensample them
    for f in glob.glob(os.path.join(folderIn, imageExtension)):

        # Read image
        Iin = cv2.imread(f,0)
        row = Iin.shape[0]
        col = Iin.shape[1]

        # Downsample image
        newSize = (int(round(col / factor)), int(round(row / factor)))
        Iout = cv2.resize(Iin,newSize)

        # Save image
        fileName = ntpath.basename(f)
        imageFilePath = folderOut + "\\" + fileName
        print imageFilePath
        cv2.imwrite(imageFilePath,Iout)
