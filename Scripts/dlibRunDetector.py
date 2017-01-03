import glob
import os
import sys

import dlib
from skimage import io

from DlibObjectDetection.MyMachineeLearning.Code.DetectionXML import DetectionXML

# Params
detectorPath = sys.argv[1]
testing_folder = sys.argv[2]
xmlOutput = sys.argv[3]

# Create an xml detection file
xmlDetection = DetectionXML(xmlOutput,0)

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector(detectorPath)


# # We can look at the HOG filter we learned.  It should look like a face.  Neat!
# win_det = dlib.image_window()
# win_det.set_image(detector)

# Now let's run the detector over the images in the faces folder and display the
# results.
for f in glob.glob(os.path.join(testing_folder, "*.bmp")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img)

    print("Number of wheels detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    # Save detection
    if(len(dets) > 0):
        xmlDetection.addImage(f, dets)

# Export file
xmlDetection.exportXML()