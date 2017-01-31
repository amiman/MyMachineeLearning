import os

from DlibObjectDetection.MyMachineeLearning.Code import DataManipulter

inputFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\CropDetections")
outFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\CropDetectionsOptimalSize")

# Create output folder if it dosen't exists
try:
    os.mkdir(outFolder)
except:
    print("Folder allready exsits")

dataManipulater = DataManipulter.DataManipulter()

dataManipulater.ResizeDataToOptimalSize(inputFolder,outFolder,"*.bmp")

