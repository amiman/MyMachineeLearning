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

def CropImageAccodringToDetections(folderOut, inputXmlPath, outputXmlPath):

    # Go over the xml file and crop images accroding to detections
    xmlIn = DetectionXML.DetectionXML(inputXmlPath, 1)
    xmlOut = DetectionXML.DetectionXML(outputXmlPath, 0)

    # Get the detection boxes
    frameData = xmlIn.getImageDetections()

    # Go over each detection and find the boudnog box for all detections
    minTop = 9999
    minLeft = 9999
    maxRight = -1
    maxBottom = -1
    for frame in frameData:
        cropSampledDetections = dlib.rectangles()

        if (len(frame.GetDetections()) == 1):

            print frame.GetFilePath()
            img = cv2.imread(frame.GetFilePath())

            for box in frame.GetDetections():

                # We have one box crop a predefined area around it
                bufferY = 240 - int(round(box.height()/2))
                bufferX = 320 - int(round(box.width() / 2))

                imgCrop = img[box.top()-bufferY:box.top()+bufferY,box.left()-bufferX:box.left()+bufferX]

                # Save image
                fileName = ntpath.basename(f)
                imageFilePath = folderOut + "\\" + fileName
                print imageFilePath
                cv2.imwrite(imageFilePath, imgCrop)

                newBox = dlib.rectangle(box.left() - bufferX, box.top() - bufferY, box.left() + bufferX,
                                            box.top() + bufferY)

                cropSampledDetections.append(newBox)


        else:
            print frame.GetFilePath()
            img = cv2.imread(frame.GetFilePath())
            for box in frame.GetDetections():

                if(box.top() < minTop):
                    minTop = box.top()

                if(box.left() < minLeft):
                    minLeft = box.left()

                if(box.right() > maxRight):
                    maxRight = box.right()

                if(box.bottom() > maxBottom):
                    maxBottom = box.bottom()

            # After we found the total bounding box crop the input image acroding to it and update boxes
            imgCrop = img[minTop:maxBottom,minLeft:maxRight]

            # Save image
            fileName = frame.GetFrameBaseName()
            imageFilePath = folderOut + "\\" + fileName
            print imageFilePath
            cv2.imwrite(imageFilePath, imgCrop)

            for box in frame.GetDetections():

                newBox = dlib.rectangle(box.left() - minLeft,box.top() - minTop,box.left() - minLeft + box.width(),box.top() - minTop + box.height())

                cropSampledDetections.append(newBox)

        # Add downsampled detections to a new file
        xmlOut.addImage(imageFilePath, cropSampledDetections )

    # Save the new downsampled xml file
    xmlOut.exportXML()
