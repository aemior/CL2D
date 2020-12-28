# CL2D
Source Code for CL2D
## requirement
Test in ubuntu 20.04 and Python 3.8.5
1. opencv 3.4.5 for linemodecore (compile with opencv_world).
  - if don't work please recompile linemodecore:change THR in CMakelists.txt,50 for evaluate,60 for visualize
  ```
  cd linemodecore
  mkdir build
  cd build
  cmake ..
  make
  ```
  - copy linemodecore.so to ../libs


2. pip3 install opencv-python shapely

## Papare data
Extract [YNU-BBD2020.zip](https://drive.google.com/file/d/19o_row_RqR1y5bNN1lbmgtauSzs6h5xB/view?usp=sharing) to ./

## Visualize
To show Ground Truth
```
python3 test.py -r show_gt
```
To test CL2D in YNU-BBD2020
```
python3 test.py -r detect
```

## Modeling
```
python3 test.py -r modeling
```

## Evaluate
CL2D-HSV,CL2D_RGB,LineMod:
```
python3 evaluate.py -m HSV 
python3 evaluate.py -m RGB
python3 evaluate.py -m LM
```

