# EdgeAIContest3
This repository is to work on the [The 3rd AI Edge Contest](https://signate.jp/competitions/256).

## Structure
    .
    ├── src             # Source files (pre/post/processing)
    ├── test            # Testing files (small unit tests etc.)
    ├── model           # Models (binary/json model files)
    ├── data            # Data (augmented/raw/processed)
    ├── notebook        # Jupyter Notebooks (exploration/modelling/evaluation)
    ├── LICENSE         
    └── README.md       

## Setup data
Please download data from the competition into the ```data/``` folder:

*After setting the [signate CLI link](https://pypi.org/project/signate/)*
```
cd data/
signate download --competition-id=256
```

## Object detection

### Retinanet
The keras implementation of retinanet can be found [here](https://github.com/fizyr/keras-retinanet), An explication of the paper is accessible [here by Alisher](https://github.com/alisher0717/machine-learning-notes/blob/master/object-detection-papers/RetinaNet.pdf)

### Yolov4

The tensorflow implementation of yolov4 can be found [here](https://github.com/sicara/tf2-yolov4).
Tested with : ```tensorflow=2.2.0```.

We train the network with the [C version](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects),
and then convert the weights into ```.h5```.

## Tracking
Implementation based on [MOT paper](https://paperswithcode.com/paper/a-simple-baseline-for-multi-object-tracking).


## Submission