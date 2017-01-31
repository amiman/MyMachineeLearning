import os
import DlibObjectDetection.MyMachineeLearning.Code.Algorithm

# Params
# TrainFolders = ["C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\TrainingImages",
#                 "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In",
#                 "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In"]

TrainFolders = [
                "C:\_work\Data\SpecficTestCases\Cars\DlibObjectDetector\TrainingImages",
                "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In",
                "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161130_150601\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_113842\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114110\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_120424\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161111_092018\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161111_095432\In",
                # "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-ON_880_20160713\\133131\In",
                    ]

InputImageDirectories = TrainFolders[1:len(TrainFolders)]

index = 0
for TrainFolder in TrainFolders:

    if index >= len(InputImageDirectories):
        break

    TrainFolder = os.path.normpath(TrainFolder)
    TestFolder = os.path.normpath("C:\_work\Data\SpecficTestCases\Cars\DlibObjectDetector\TestingImages")
    TrainXMLFile = "training.xml"
    ImageFileExtension = ".bmp"

    InputImageDirectory = os.path.normpath(InputImageDirectories[index])
    OutputPath = os.path.normpath("C:\_work\Data\SpecficTestCases\Cars\DlibObjectDetector\Output")

    ObjectSVMOutput = OutputPath + "\detector.svm"
    DetectionOutput = InputImageDirectory + "\\training.xml"
    DetectionTag = "box"

    algo = DlibObjectDetection.MyMachineeLearning.Code.Algorithm.Algorithm(True, TrainFolder, TestFolder, TrainXMLFile, OutputPath, InputImageDirectory, ImageFileExtension, False)

    algo.Run()

    index += 1