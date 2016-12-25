from lxml import etree as ET

class DetectionXML(object):

    def __init__(self, path):
        self.path = path
        self.root = ET.Element("dataset")
        ET.SubElement(self.root, "name").text = "Detection data"
        self.images = ET.SubElement(self.root, "images")

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





