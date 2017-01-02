import ntpath
import os

class FrameData(object):

    def __init__(self, path, detections):

        self.path = os.path.normpath(path)
        self.baseName = ntpath.basename(path)
        self.detections = detections

    def AddDetection(self, detection):

        self.detections.append = detection

    def AddDetections(self, detections):

        for a, boundingBox in enumerate(detections):
            self.AddDetection(boundingBox)

    def GetFrameBaseName(self):
        return self.baseName

    def GetDetections(self):
        return self.detections

    def GetFilePath(self):
        return self.path



