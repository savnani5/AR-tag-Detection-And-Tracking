# AR-tag-Detection-And-Tracking
This project will focus on detecting a custom AR Tag (a form of fiducial marker),
that is used for obtaining a point of reference in the real world, such as in
augmented reality applications. There are two aspects to using an AR Tag, namely
detection and tracking, both of which will be implemented in this project. The
detection stage will involve finding the AR Tag from a given image sequence while
the tracking stage will involve keeping the tag in “view” throughout the sequence
and performing image processing operations based on the tag’s orientation and
position (a.k.a. the pose).

Checkout [this report]() for detialed explannation of detection, tracking of the tag and superimposing over it.

## Input

[Input Data](https://drive.google.com/drive/folders/1b_cSKQp5dlNqVjAsJskwU_5_8V1B5Uq1?usp=sharing)

## Output

[Output Data](https://drive.google.com/drive/folders/19yLQtRxngrrmcS1Lgx9lMOd856wHkDdb?usp=sharing)

### Detection 

**Warped AR tages**

![eg1](git_images/tag1.png)   ![eg2](git_images/tag2.png)

### Tracking and Superimposing

![testudo](git_images/testimg.png)     ![testudo](git_images/testudo.gif)

![testudo](git_images/testimg2.pmg)     ![eg1](git_images/cube.gif)

## How to Run the Code

1) Change the path of the input videos in the video list

```video_list = ['Tag0.mp4', 'Tag1.mp4', 'Tag2.mp4', 'multipleTags.mp4']```

2) Run the file *Two_a.py* for the image superimpostion.

3) Run the file *Two_b.py* for the cube superimposition.


## References
1) https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
2) https://www.geogebra.org/?lang=en
3) https://math.berkeley.edu/~hutching/teach/54-2017/svd-notes.pdf
4) https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html#table-of-content-contours
5) https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
6) ENPM 673, Robotics Perception - Theory behind Homography Estimation
7) CSCE 441: Computer Graphics - Image Warping - Jinxiang Chai
8) https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/


