import os
import glob
import ntpath

import cv2

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
            newImagePath = outFolder + "\\" + "name"

            # Reshape image to optimal size
            self.ReshapeImage(optimalRow,optimalCol, Iin, newImagePath)

    def ReshapeImage(self, newSizeRow, newSizeCol, inputImage, outputPath):

        # Find what is the relation between the sei of the image and the new size
        row = inputImage.shape[0]
        col = inputImage.shape[1]

        rowRatio = row / newSizeRow
        colRatio = col / newSizeCol

        # Reshape the image according to the bigger ratio
        maxRatio = max(rowRatio,colRatio)
        resizeImage = cv2.resize(inputImage,((row*maxRatio),(col*maxRatio)))

        # Crop the image to the optimal size
        centerPointX = newSizeRow / 2
        centerPointY = newSizeCol / 2

        cropResizeImage = resizeImage[centerPointX-centerPointX:centerPointX+centerPointX,centerPointY-centerPointY:centerPointY+centerPointY]


