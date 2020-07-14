# Introduction (to be filled)

introduce MLT

MLT's ranking, prize we have won

mention contest details (who organized the contest, total number of participants, number of gold/silver/bronze medal receivers and winners, etc)


# Edge AI contest task definition (to be filled)
problem definition

training/test data

number of images per class

evaluation metric - MOTA




# How did we tackle the task?
We discussed and decided to divide the main task into sub-tasks. The primery aim was to be able to work on different sub-tasks independently, i.e. not sequential - we should not have to wait to finish a subtask before starting the next one. According to these assumptions, we defined two sub-tasks:

- object detection sub-task: the training videos are used as training data and performance is evaluated considering only object detection performance - mAP (mean Average Precision, Average Precision per class)

- object tracker sub-task: although the final evaluation would consider the object detector and object tracker's performance, we decided to use the ground truth detection labels (bbox coordinates of objects) to develop object tracker.



## 1. Pre-processing (to be filled)

data split: train/validation

split the video into frames, etc

## 2. Object Detection
There are mainly two types of deep learning based object detection algorithms: 

- two-stage obejct detectors: R-CNN, Fast R-CNN, Faster R-CNN 

- single-stage/single-shot object detectors: SSD, YOLO, RetinaNet. 


Because single-stage object detectors are faster and since the contest was about edge AI, we decided to use single-stage object detector - RetinaNet. We have chosen RetinaNet detector because we already had an experience on training RetinaNet model. 

First, we have trained the model using all available classes in training data (Pedestrian, Car, Truck, Signs, Svehicle, Bus, Train, Motorbike, Signal, Bicycle) and fine tuned on only two classes (Pedestrian and Car) since only those classes would be considered during the evaluation. Then, we thought it could be better to train on corresponding classes (Pedestrian and Car) to get rid of unnecessary class predictions (which could make it easier for detector to predict correct classes). We have confirmed that this works better than the former one in terms of mAP score on validation set. Pedestrian: 0.7713 AP & 
Car: 0.9244 AP.

After a comprehensive visual analysis, we have seen that the model confuses some classes. For instance, the model predicts the "Bus" as "Car". This kind of mis-detections increase the False Positive (which leads to lower MOTA metric).  Then, we concluded that it may be better to include the similar classes on top of previous classes: [Pedestrian, Car] + [Bus, Truck, Svehicle]. The newly added classes were confused with Pedestrian class. By including (and re-training the detection model) we prevent those confusions. 

Training details: 
RetinaNet model with ResNet101 backbone (pre-trained on ImageNet)

Didn't resize the video frames (still images) which had the shape of `1936x1216x3`. 

Learning rate: `1e-5` 
Augmentations: rotation `(-0.2, 0.2)`, translation `(-0.2, 0.2)`, shear `(-0.2, 0.2)`, scaling `(0.7, 1.4)`, horizontal flip (with `0.5` probability).

Classes: `Pedestrian`, `Car`, `Truck`, `Bus`, `Svehicle`

Trained for 100 epochs. Epoch 15 snapshot has been chosen (we didn't consider the small mAP score differences between different snapshots, because doing this may lead us to overfit on validation dataset). Epoch 15 had AP (average precision) for "Pedestrian": `0.7713` and for "Car": `0.9244`.

Training data format: we have used csv data format: `image_fpath,x1,y1,x2,y2,class` (please check keras-retinanet repository).




## 3. Object Tracking (to be filled)




## 4. Combining object detection and object tracking - prediction, post-processing.
Test time augmentations: during the inference instead of feeding the original image only, we fed the batch of images: original image + augmented versions. As for the test time augmentation we have tried "horizontal flip", "brightening: brighten the image", "darken: darken the image", "right/left crop: crop the right/left region of an image". Among these test time augmentations we find only horizontal flip helpful. We could not exploit all of them because of restricted inference time per frame.
Confidence scores: we have fixed the confidence threshold as 0.5 for original image predictions. However, for flipped version (basically for every test time augmentations we have tried) of the image, we defined a new parameter - conf_score_bias=0.1 -  which is added to 0.5 to yield a confidence threshold for predictions from flipped version of input image. This parameter (conf_score_bias) implies that we should not give similar chance for original and augmented image prediction, i.e. accept the predictions from augmented version if and only if it is bigger than 0.6 which means it is more confident.
Combining predictions for different image versions: since we have fed 2 images (original + horizontally flipped) to the model, we get multiple predictions, i.e. we had to merge object bounding boxes. We have used two different nms methods: (we call it) local nms and global nms. Local nms is applied to predictions got from different batch images. Global nms is applied after merging (concatenating) the predictions from original and flipped input images. For local nms we have used IoU=0.8 and for global nms we have used IoU=0.5.
post-processing heuristic: during the visual analysis, we found that there are no objects at very top and very bottom part of frame at all. Thus, we have applied this heuristic as a post-processing; discard any predictions if y-coordinate was above 365 or below 851.



# What we have tried and didn't work?
Test time augmentations:  "brightening", "darkening", "right crop", "left crop" augmentations did not work well. Only "horizontal flip" helped to increase the overall performance.
Adaptive nms threshold for "Pedestrian" class: use adaptive nms IoU threshold according to the number of detections. This was due to crowded scenes, i.e., if there are a lot of pedestrian in the frame then probably their bounding boxes should be overlapping more compared to less crowded frames. We tried this, but could not fine-tune well to get better MOTA score.
Receject confidence score outliers: we had a plan to reject object outliers according to their confidence score considering that the objects in the same frame should be visible with similar probability. However, this didn't help either because we did not have enough time to fine-tune or because it really does not help.
Classification phase: we considered to train another model to classify the object detection predictions into "Pedestrian" and "Car" classes (binary classification). This could help to curate object detection results as well as object tracker when matching inter-frame objects. However, we did not have enough time to try this. 
… some failed trials form object tracker can be added here …



# Conclusion
… conclusion (less submission could help us not to overfit to public score)…






References: 
https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html
