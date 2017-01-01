import os
import sys

import dlib

# The idea of learning from a small data set is the following:
# 1. Use a simple linear classifier that can be trained using small data set and give reasonable results
# 2. Use the trained classifier to detect the object in a video
# 3. Start tracking the detected object across frames and add the new learned examples to a new data set
# 4. Retrain the classifier again with the larger data set and do stages 1-3.
# 5. After exhausting stages 1-3 do data augmentation involving light,noise and more manipulations
# 6. When there is enough data move fom simple linear classifier to training using deep learning


################################################## 1. Learn simpel linear classsfier #############################################################

# Params
train_folder = sys.argv[1]
test_folder = sys.argv[2]

# Do trainnig using dlib libary. We can use any other algorithm to do our initial training.
options = dlib.simple_object_detector_training_options()

# For symmetirc objects
options.add_left_right_image_flips = True

# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5

# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 4
options.be_verbose = True

training_xml_path = os.path.join(train_folder, "training.xml")
testing_xml_path = os.path.join(test_folder, "testing.xml")

# This function does the actual training.  It will save the final detector to
# detector.svm.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the face boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file included with
# dlib.
dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

################################################## 2. Detect and track objects in videos (stages 2- 3) #############################################################
