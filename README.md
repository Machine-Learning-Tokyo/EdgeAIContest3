# The 3rd AI Edge Contest (Algorithm Contest 2)
This repository present **MLT Team** solution for the  [The 3rd AI Edge Contest](https://signate.jp/competitions/256).

![object_tracking](https://github.com/Machine-Learning-Tokyo/EdgeAIContest3/blob/master/imgs/object_tracking.gif)

## Introduction
First of all thank you to [Signate](https://signate.jp/) for hosting this exciting competition.
Our team MLT is based in Tokyo and we are really interested in the Edge devices and AI applications at the edge.
Having the opportunity to work on **“a made in” Tokyo dataset** is really motivating and gives the feeling to work on a real world project compared to other competitions.
On top of that, the field of [MoT](https://en.wikipedia.org/wiki/Multiple_object_tracking) is still a hot research topic, and certainly very challenging by it’s complexity and application.
Also the sense of working together with one aim to get a good rank was a continuous source of motivation.

We are planning to continue this project and actually deploying the code on an edge device.

## Structure
    .
    ├── src             # Source files (submission file / src module / training)
    ├── evaluation      # Testing files (small unit tests etc.)
    ├── model           # Models (binary/json model files)
    ├── data            # Data (augmented/raw/processed)
    ├── notebook        # Jupyter Notebooks (Data exploration/ Data preparation)
    ├── LICENSE
    └── README.md

## Solution overview

Our solution is composed of two sequential modules: Object detection and Tracking.

![overview](https://github.com/Machine-Learning-Tokyo/EdgeAIContest3/blob/master/notebook/overview.png)

### 1] Object Detection
The first step consists of detecting objects in the frame.
We selected RetinaNet with ResNet 101 as CNN backbone as our model architecture because it is really efficient with dense and small scale objects while assuring fast inference for edge device application as being a single stage detector. Please have a look at [the annotated paper](https://github.com/Machine-Learning-Tokyo/papers-with-annotations/blob/master/object-detection/RetinaNet.pdf) from one of our team member to get a detailed understanding of this particular model.

One specification was to adopt a **batching augmentation**, that consist of inferring the detection on augmented image, and finally applying Non-Maximum-Suppression on all the detected objects. We tried out several augmentations (Dark-Bright / Crop Side / ...), but based on experiment our final submission only use *flip Right-Left* augmentation.

Finally, we did some heuristics bounding box filtering based on our dataset exploration (final submission use filtering based on image position.)

All our training related information is summarized in [ObjectDetectionTraining.md](src/ObjectDetectionTraining.md).

### 2] Tracking

We formalized the tracking problem as a maximum weighted matching problem for objects in two adjacent frames and solved it using [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm#:~:text=Hungarian%20algorithm%20%2D%20Wikipedia-,Hungarian%20algorithm,anticipated%20later%20primal%2Ddual%20methods.).

For matching costs, we utilized several features as:
- position
- size
- image similarity (histogram)

Our tracker keeps all the history of object tracking and estimates the next position of each object in the next frame by linear or quadratic regression.

Then, the tracker tries to match objects with close position, similar size and similar image as much as possible.
During object matching, the tracker also takes into account object appearance and disappearance.

We added some virtual objects where the objects matched to these virtual objects are regarded as newly appeared or disappeared.

The disappeared objects are also kept in the tracker for a while and can be matched to some objects in the subsequent frames.

## Improvement and Lesson learned

### Improvements
There are lot of room for improvement:
- Better tunned or new batch augmentation.
- Pedestrian classifier to reduce FP. *(developed but not tuned enough to be used for the submission)*
- Automation pipeline for parameters tunning.

### What did we learn
- Public score does not always reflect the private score, and those make the tunning hard.
- Running dummy inference in the load_model method allows us to reduce inference time.
- Heuristics are very valuable to increase the score
- We should have spent some resource on cleaning the dataset
- When you think you're done with data exploration, you're not: **Make summary and clean report.**

## Data
Please download data from the competition into the ```data/``` folder:

**Note that Signate may not open the dataset.**

After setting the [signate CLI link](https://pypi.org/project/signate/)
```
cd data/
signate download --competition-id=256
```

## Notebook
This folder contain our **DataExploration notebook**:

![overview](https://github.com/Machine-Learning-Tokyo/EdgeAIContest3/blob/master/notebook/all_video.png)

## Source code
Please refer to [src/README.md](src/README.md) for explanation of our source code.

## Submission - Evaluation

Evaluation contains Signate code to run local evaluation.

Run the following command to generate the sample_submit folder.
```
bash generate_mlt_submission.sh
```

In order to test submission instance run following:
```
bash test_submit.sh
cd sample_submit; pwd
python src/main.py
```
