import os

from DlibObjectDetection.MyMachineeLearning.Code import DataManipulter

inputFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\NegativeExamples")
outFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\NegativePatches")

# Create output folder if it dosen't exists
try:
    os.mkdir(outFolder)
except:
    print("Folder allready exsits")

dataManipulater = DataManipulter.DataManipulter()

dataManipulater.CrateNegativeExamples(inputFolder,outFolder,"*.bmp",102,80,1000)
