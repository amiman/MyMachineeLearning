import os
import sys
import glob

import dlib
from skimage import io


# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.
if len(sys.argv) != 3:
    print(
        "Give the path to the examples/faces directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py ../examples/faces")
    exit()
train_folder = sys.argv[1]
test_folder = sys.argv[2]


# Now let's do the training.  The train_simple_object_detector() function has a
# bunch of options, all of which come with reasonable default values.  The next
# few lines goes over some of these options.
options = dlib.simple_object_detector_training_options()
# Since faces are left/right symmetric we can tell the trainer to train a
# symmetric detector.  This helps it get the most value out of the training
# data.
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
dlib.train_simple_object_detector(training_xml_path, "detector.svm",options )



# Now that we have a face detector we can test it.  The first statement tests
# it on the training data.  It will print(the precision, recall, and then)
# average precision.
# print("")  # Print blank line to create gap from previous output
# print("Training accuracy: {}".format(
#     dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
# # However, to get an idea if it really worked without overfitting we need to
# # run it on images it wasn't trained on.  The next line does this.  Happily, we
# # see that the object detector works perfectly on the testing images.
# print("Testing accuracy: {}".format(
#     dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))





#
#
#
#
#
#
#
# # Finally, note that you don't have to use the XML based input to
# # train_simple_object_detector().  If you have already loaded your training
# # images and bounding boxes for the objects then you can call it as shown
# # below.
#
# # You just need to put your images into a list.
# images = [io.imread(faces_folder + '/2008_002506.jpg'),
#           io.imread(faces_folder + '/2009_004587.jpg')]
# # Then for each image you make a list of rectangles which give the pixel
# # locations of the edges of the boxes.
# boxes_img1 = ([dlib.rectangle(left=329, top=78, right=437, bottom=186),
#                dlib.rectangle(left=224, top=95, right=314, bottom=185),
#                dlib.rectangle(left=125, top=65, right=214, bottom=155)])
# boxes_img2 = ([dlib.rectangle(left=154, top=46, right=228, bottom=121),
#                dlib.rectangle(left=266, top=280, right=328, bottom=342)])
# # And then you aggregate those lists of boxes into one big list and then call
# # train_simple_object_detector().
# boxes = [boxes_img1, boxes_img2]
#
# detector2 = dlib.train_simple_object_detector(images, boxes, options)
# # We could save this detector to disk by uncommenting the following.
# #detector2.save('detector2.svm')
#
# # Now let's look at its HOG filter!
# win_det.set_image(detector2)
# dlib.hit_enter_to_continue()
#
# # Note that you don't have to use the XML based input to
# # test_simple_object_detector().  If you have already loaded your training
# # images and bounding boxes for the objects then you can call it as shown
# # below.
# print("\nTraining accuracy: {}".format(
#     dlib.test_simple_object_detector(images, boxes, detector2)))