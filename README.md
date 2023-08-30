# A longitudinal multi-task learning algorithm for Alzheimer's disease considering time accumulative effects(TA_MTL)

Paper title:
**A longitudinal multi-task learning algorithm for Alzheimer's disease considering time accumulative effects**

Copyright (C) 2023
Hong Yu
Chongqing University of Posts and Telecommunications

## Usage

- Python 3.8
- Numpy, pytorch, sklearn, random, math, matplotlib

## Describe

The repositories include two folders. 

1. The folder  “code” provides the source codes of our TA_MTL model, the reproduced comparison model and tools needed to run these models. You will need to modify the internal code to achieve specific application effects.
2. The folder  “src” provides ADNI1:Complete 1Yr 3T dataset after processing, The original data could not be provided because of the requirements of the data use agreement. Our processing included image processing and brain region segmentation using FreeSurfer(7.3.0), data complementation and normalization.
3. The corresponding paper has been submitted to IEEE Journal of Biomedical and Health Informatics. The complete code will be released after the paper is accepted. 

# Experimental environment:

Vscode 1.81.1, Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz   2.90 GHz

# Virtual environment:

python 3.8, numpy 1.21.5, pytorch 1.13.1, sklearn 1.1.1, matplotlib 3.5.2
