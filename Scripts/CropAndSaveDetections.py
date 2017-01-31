from DlibObjectDetection.MyMachineeLearning.Code import Util

files = [
    "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_114516\In\\training.xml",
    "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161201_115848\In\\training.xml",
    "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\A4\IN-AN_7822_20161130_150601\In\\training.xml",
    "C:\_work\Data\SpecficTestCases\SwearLids\AUDI\IN-AN_7822_20161110_113842\In\\training.xml",

]

OutFolder = "C:\_work\Data\SpecficTestCases\Wheels\DlibObjectDetector\CropDetections"

# Go over all the detections files
for xmlFile in files:

    Util.CropDetections(OutFolder,xmlFile)
