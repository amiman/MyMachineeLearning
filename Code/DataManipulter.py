import os
import glob
import ntpath

import cv2
import numpy as np

class DataManipulter(object):

    def __init__(self):

        self.numOfImage = 0
        self.totalSizeRow = 0
        self.totalSizeCol = 0

    def ResizeDataToOptimalSize(self, inFolder, outFolder, imageExtension):

        # Go over all the files in the inFolder and find the optimal size to resacle all the data to
        for imagePath in glob.glob(os.path.join(inFolder, imageExtension)):

            # Read image
            Iin = cv2.imread(imagePath, 0)

            # Find the size of the image
            self.totalSizeRow += Iin.shape[0]
            self.totalSizeCol += Iin.shape[1]

            self.numOfImage += 1

        # The optimal size is the mean of the rows ans cols
        optimalRow = (int)(self.totalSizeRow / self.numOfImage)
        optimalCol = (int)(self.totalSizeCol / self.numOfImage)

        # Now go over the images again and reshape them according to the optimal size
        for imagePath in glob.glob(os.path.join(inFolder, imageExtension)):

            # Read image
            Iin = cv2.imread(imagePath, 0)

            # new image name
            name = ntpath.basename(imagePath)
            newImagePath = outFolder + "\\" + name

            # Reshape image to optimal size
            cropResizeImage = self.ReshapeImage(optimalRow,optimalCol, Iin, newImagePath)

            # Save the reshape image
            cv2.imwrite(newImagePath,cropResizeImage)

    def ReshapeImage(self, newSizeRow, newSizeCol, inputImage, outputPath):

        # Find what is the relation between the sei of the image and the new size
        row = inputImage.shape[0]
        col = inputImage.shape[1]

        rowRatio = float(newSizeRow)/float(row)
        colRatio = float(newSizeCol)/float(col)

        # Reshape the image according to the bigger ratio
        maxRatio = max(rowRatio,colRatio)
        resizeImage = cv2.resize(inputImage,None, fx=maxRatio, fy=maxRatio, interpolation = cv2.INTER_CUBIC)

        # Crop the image to the optimal size
        centerPointX = newSizeRow / 2
        centerPointY = newSizeCol / 2

        cropResizeImage = resizeImage[centerPointX-centerPointX:centerPointX+centerPointX,centerPointY-centerPointY:centerPointY+centerPointY]

        return  cropResizeImage

    ######## NOTE: The assumption is that in the images there ARN'T ANY true postive objects!!!!!!!!!!!!!!!!!!!!!! #################################
    def CrateNegativeExamples(self, inFolder, outFolder,imageExtension, patchRow, pathCol, numberOfNoneObjectPerImage):

        stepX = int(pathCol / 2)
        stepY = int(patchRow / 2)

        # Go over all the images in the workingDirectory and crop areas for false postives
        for imagePath in glob.glob(os.path.join(inFolder, imageExtension)):

            # We have a valid file open it for processing
            currentFrame = cv2.imread(imagePath, 0)

            row = int(currentFrame.shape[0])
            col = int(currentFrame.shape[1])

            # Create numberOfNoneObjectPerImage for the image
            for index in range(numberOfNoneObjectPerImage):
                # Create a random point to crop around
                pointx = int(
                    np.random.uniform(pathCol, col - pathCol))
                pointy = int(
                    np.random.uniform(patchRow, row - patchRow))

                # Crop patch
                patch = currentFrame[pointy - stepY:pointy + stepY, pointx - stepX:pointx + stepX]

                # Save image
                filePath = outFolder + "//" + os.path.splitext(ntpath.basename(imagePath))[0] + "_" + str(index) + '.jpg'
                cv2.imwrite(filePath, patch)



