import os
import glob
import ntpath

class ImageDirectory(object):

    def __init__(self, path, imageFileType):

        self.path = path
        self.fileType = imageFileType

        self.filesPathsToIndexes = {}
        self.filesIndexesToPaths = []

        self.filesPaths = []

        self.numOfFiles = 0

        self.ReadDirectory()

    def ReadDirectory(self):

        # Go over the directory and create a data structure to get through the files
        index = 0
        for filePath in glob.glob(os.path.join(self.path,"*" + self.fileType)):

            self.filesPaths.append(filePath)
            self.filesIndexesToPaths.append(ntpath.basename(filePath))
            self.filesPathsToIndexes[ntpath.basename(filePath)] = index

            index += 1


        self.numOfFiles = index

    def GetNextFileByName(self, fileName):

        # Find the current file index
        currentIndex = self.filesPathsToIndexes.get(fileName)

        # Return next file name
        nextIndex = currentIndex + 1
        return self.filesIndexesToPaths(nextIndex)

    def GetFileIndexByName(self, fileName):

        return self.filesPathsToIndexes.get(fileName)

    def GetFilesPathUsingIndexes(self, startIndex, endIndex):

        return self.filesPaths[startIndex:endIndex]
