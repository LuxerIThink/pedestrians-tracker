# Pedestrians Tracker

This Python project is designed to track pedestrians
in videos using preprocessed frames and ready bounding boxes.
It utilizes the Market-1501 dataset,
which has been modified for this purpose.

This project was developed for the
Artificial Intelligence in Robotics course
at Poznan University of Technology in Poland.

## How it works

### Usage

The project can be run using the following command:
```bash
python3 main.py <dataset_folder> <optional_output_file>
```

It need Python 3.10 or higher and libraries
specified in requirements.txt to run.

### Input

The input is folder containing
- "frames" folder with images,
- "bboxes" folder with bounding boxes

Example input structure:
```
test_dataset
├── bboxes.txt
└── frames
    ├── c6s1_000501.jpg
    ├── c6s1_001426.jpg
    └── c6s1_001451.jpg
```

Example bboxes.txt file:
```
c6s1_000451.jpg
1
420.836933 144.188985 88.328294 216.466523
c6s1_000476.jpg
3
325.044276 151.653348 126.894168 204.025918
177.001080 160.361771 90.816415 153.019438
129.726782 129.260259 83.352052 195.317495
```

Where for every photo is a template:
```
photo name in frames folder
numbers of bounding boxes
x1 y1 x2 y2 of first bounding box
x1 y1 x2 y2 of second bounding box (if exists)
... 
```

### Compare clippings:

Three methods are employed for comparison:

Matching images using OpenCV's matchTemplate function with the CCOEF normed method.
Comparing gray histograms.
Calculating the Intersection over Union (IoU).
Each method produces a value between 0 and 1.
These values are averaged by summing them up and dividing by 3,
resulting in a final value between 0 and 1.

### Determine the most probable connection:

A probability matrix is created, which represents the probability
of a connection between each pair of current and previous indexes.
The columns correspond to the current indexes,
while the rows represent the previous indexes.
The most probable connection for each previous index is determined
using the SciPy Linear Sum Assignment algorithm.

If the probability is too low, it is set to 0.0.
If the sum of probabilities for a given previous index is also 0.0,
the index is assigned -1, indicating a new element. Otherwise,
the index is assigned as "n," representing the nth previous element.

### Output

Example output file:
```
-1 0
-1 0 1
0 2 -1
1 2 -1 0
1 -1 0 -1 3 2
4 0 1 3 2
1 0 2
1 0
```

## Tests

The provided dataset can be accessed
[dataset](https://drive.google.com/file/d/1saVmRWqBBfeJTLH3Lo91bXtbI2h4T3vi/view)

The accuracy achieved in the tests is 85.25%.
