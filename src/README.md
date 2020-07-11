# Source Explained

For the sake of fast development and submissions, the current module for object detection is not yet updated. Please refer to prediction.py to get the latest developement.

## Content
Composed by 1 main process: ```process_video.py```, that takes care of loading the video, setting up the **detection_module**, the **submission_helper**, and **tracking_module**.

- prediction.py: the submission inference src.

- Detection module: is a wrapper of tensorflow implementation, that generate formated prediction.

- Tracking module: generate tracking based on previous-current frame transformation and detection output.

- Submission helper: generate a submission **.json** file.

- Training label generation.py

- Other experimental scripts...

## Tracker
#### object_tracker.py

Track objects and assign IDs.

#### stabilizer.py (under development)

Stabilize the camera motion for two adjacent frames and transform the first frame toward the second frame.

### Run

```python
from object_tracker import Tracker

...
image_size = (1936, 1216, frame)

tracker = Tracker(image_size)
...

# input prediction data for each frame in order
# input: {'box2d': [x1, y1, x2, y1]}
# output: {'id': id, box2d': [x1, y1, x2, y1], 'mv': [vx, vy], 'scale': [sx, sy], 'occlusion': number_of_occlusions}
prediction = tracker.assign_ids(prediction)
...
```

### Test

```bash
python3 object_tracker.py --input /path/to/annotation/directory --output /path/to/output.json

Frame #1
    #Boxes: Car=8, Pedestrian=12
    Execution time: total=0.00034165, max=0.00034165
    Total cost:  2e+16
Frame #2
    #Boxes: Car=7, Pedestrian=14
    Execution time: total=0.00176668, max=0.00176668
    Total cost:  2.0000000000000004e+16
...

Frame #600
    #Boxes: Car=5, Pedestrian=14
    Execution time: total=0.00174618, max=0.78549790
    Total cost:  2.43e+18
Overall (../train_annotations/train_02.json)
    Car: total=2984, sw=221, tp=2569, err=0.07406166
    Pedestrian: total=3072, sw=695, tp=1856, err=0.22623698
    All: err=0.15125495
```