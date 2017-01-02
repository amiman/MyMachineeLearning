from lxml import etree as ET

import dlib

import FrameData
from DlibObjectDetection.MyMachineeLearning.Code.FrameData import FrameData


class DetectionXML(object):

    IMAGE_PATH_ATTRBIUTE =                  'file'
    IMAGE_BOUNDING_BOX_FIELD =              'box'
    IMAGE_BOUNDING_BOX_ATTRIBUTE_TOP =      'top'
    IMAGE_BOUNDING_BOX_ATTRIBUTE_BOTTTOM =  'bottom'
    IMAGE_BOUNDING_BOX_ATTRIBUTE_HEIGHT =   'height'
    IMAGE_BOUNDING_BOX_ATTRIBUTE_LEFT =     'left'
    IMAGE_BOUNDING_BOX_ATTRIBUTE_RIGHT =    'right'
    IMAGE_BOUNDING_BOX_ATTRIBUTE_WIDTH =    'width'

    def __init__(self, path, type):

        self.path = path

        if type == 0:

            self.root = ET.Element("dataset")
            ET.SubElement(self.root, "name").text = "Detection data"
            self.images = ET.SubElement(self.root, "images")

        elif type == 1:

            tree = ET.parse(self.path)
            self.root = tree.getroot()


    def setPath(self,path):
        self.path = path

    def addImage(self, imageName, imageDetections):

        # Go over all the detection boxes and add the detection data
        image = ET.SubElement(self.images, "image", file=imageName)
        for k, d in enumerate(imageDetections):
            ET.SubElement(image, "box", top=str(d.top()), left=str(d.left()), width=str(d.width()), height=str(d.height()))

    def exportXML(self):

        tree = ET.ElementTree(self.root)
        tree.write(self.path, pretty_print=True)

    def getImageIter(self):

        return self.root.iter('image')

    def getImageDetections(self):

        frameDataList= []

        # Convert the entire tree to a list of FrameData
        index = 0
        for image in self.getImageIter():

            detections = dlib.rectangles()

            for box in image.iter('box'):

                # Convert the box filed to dlib rectangle and add to detections
                rightCorrdinate = int(box.attrib[self.IMAGE_BOUNDING_BOX_ATTRIBUTE_LEFT]) + int(box.attrib[self.IMAGE_BOUNDING_BOX_ATTRIBUTE_WIDTH])
                bottomCorrdinate = int(box.attrib[self.IMAGE_BOUNDING_BOX_ATTRIBUTE_TOP]) + int(box.attrib[self.IMAGE_BOUNDING_BOX_ATTRIBUTE_HEIGHT])
                detection = dlib.rectangle(int(box.attrib[self.IMAGE_BOUNDING_BOX_ATTRIBUTE_LEFT]), int(box.attrib[self.IMAGE_BOUNDING_BOX_ATTRIBUTE_TOP]),rightCorrdinate,bottomCorrdinate)
                detections.append(detection)

            # Crate a new frame data
            frameDataList.append(FrameData(image.attrib[self.IMAGE_PATH_ATTRBIUTE],detections))
            index += 1

        return frameDataList





