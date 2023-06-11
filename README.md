# Pedestrians Tracker

Python project to follow pedestrians on video, based
on prepared frames and bounding boxes.
For this work used Market-1501 dataset.

*Project for Artificial Intelligence in Robotics subject
on Poznan University of Technology in Poland.*

## How it works

**Compare clippings:**

There are compared with 2 methods:
- compare images with cv2 matchTemplate with ccoef normed method
- comparing grey histograms

which returns float values in 0-1 range. These values
are multiplied to obtain a single probability

**Determine the most probable connection:**

There is created probability matrix 
It contains probability in range 0-1 between each
connection where columns stands for actual indexes
and rows are previous indexes. Next there is chosen
the most probable connection for each previous index
with:

- SciPy Linear Sum Assignment

If probability is too low, there is changed to 0.0, if
sum of probabilities is also 0.0, the index is set to -1
which means new element, otherwise the index is set to n
which means nth previous element.


