import os
import Algorithm

# Params
# TrainFolders = ["C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\TrainingImages",
#                 "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In",
#                 "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In"]

TrainFolders = [
                "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In",
                "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In"]

InputImageDirectories = TrainFolders[1:len(TrainFolders)]

index = 0
for TrainFolder in TrainFolders:

    TrainFolder = os.path.normpath(TrainFolder)
    TestFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\TestingImages")
    TrainXMLFile = "training.xml"
    ImageFileExtension = ".bmp"

    InputImageDirectory = os.path.normpath(InputImageDirectories[index])
    OutputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\Output")

    ObjectSVMOutput = OutputPath + "\detector.svm"
    DetectionOutput = InputImageDirectory + "\\training.xml"
    DetectionTag = "box"

    algo = Algorithm.Algorithm(True,TrainFolder,TestFolder,TrainXMLFile,OutputPath,InputImageDirectory,ImageFileExtension)

    algo.Run()

    index += 1