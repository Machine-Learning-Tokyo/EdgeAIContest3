# RetinatNet wrapper
import tensorflow as tf
import pdb

import keras

import sys
sys.path.insert(0, '../../src/')

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import cv2
import numpy as np

# Default classes:
labels_to_names = {0: 'Pedestrian', 1: 'Car'}

class retinanet_inference():
    def __init__(self, weight_path, classe_filter=['Pedestrian', 'Car']):
        # Model
        self.weight_path = weight_path

        # Post processing parameters
        self.threshold_car = 0.6
        self.threshold_pedestrian = 0.5
        self.pedestrian_nms_thr = 0.4
        self.car_nms_thr = 0.35

        self.model = None
        self.classe_filter = classe_filter

        self.classes_list = self.CLASSES = labels_to_names
        self.label_limit = 1

        # Load model
        self.load_model()
        #self.model.summary()

    def load_model(self):
        print("Loading model: {}".format(self.weight_path))
        try:
            self.model = models.load_model(self.weight_path, backbone_name='resnet50')
        except Exception as e:
            print("Unable to load weight file: {}".format(e))
        print("   ...Done!")

    def non_max_suppression_with_scores(self, boxes, probs=None, overlapThresh=0.5):
        boxes = np.array(boxes)
        probs = np.array(probs)
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        idxs = y2

        if probs is not None:
            idxs = probs

        idxs = np.argsort(idxs)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            xx1_union = np.minimum(x1[i], x1[idxs[:last]])
            yy1_union = np.minimum(y1[i], y1[idxs[:last]])
            xx2_union = np.maximum(x2[i], x2[idxs[:last]])
            yy2_union = np.maximum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            w_union = np.maximum(0, xx2_union - xx1_union + 1)
            h_union = np.maximum(0, yy2_union - yy1_union + 1)

            # overlap = (w * h) / area[idxs[:last]]
            iou = (w * h) / (w_union * h_union)

            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlapThresh)[0])))

        return pick

    def filter_prediction(self, boxes, scores, classes_pred):
        """Clean bbox by threshold, area, NMS and format for TRACKING."""
        clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = [], [], []
        clean_bboxes_car, clean_classes_pred_car, clean_scores_car = [], [], []

        for bbox_, score_, label_ in zip(boxes.tolist(), scores.tolist(), classes_pred.tolist()):
            if label_ == -1:
                break
            if label_ == 0 and score_ < self.threshold_pedestrian:
                continue
            if label_ == 1 and score_ < self.threshold_car:
                continue
            [x1, y1, x2, y2] = bbox_
            width = x2 - x1
            height = y2 - y1

            # Cleaning too small prediction.
            if width * height < 1024:
                continue

            # Separate label for NMS
            if label_ == 0:
                clean_bboxes_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                clean_classes_pred_pedestrian.append(label_)
                clean_scores_pedestrian.append(score_)
            elif label_ == 1:
                clean_bboxes_car.append([int(x1), int(y1), int(x2), int(y2)])
                clean_classes_pred_car.append(label_)
                clean_scores_car.append(score_)
            else:
                continue

            # NMS
            pick_inds_pedestrian = self.non_max_suppression_with_scores(clean_bboxes_pedestrian, probs=clean_scores_pedestrian,
                                                                       overlapThresh=self.pedestrian_nms_thr)

            clean_bboxes_pedestrian_nms = list(clean_bboxes_pedestrian[i] for i in pick_inds_pedestrian)
            clean_classes_pred_pedestrian_nms = list(clean_classes_pred_pedestrian[i] for i in pick_inds_pedestrian)
            clean_scores_pedestrian_nms = list(clean_scores_pedestrian[i] for i in pick_inds_pedestrian)

            pick_inds_car = self.non_max_suppression_with_scores(clean_bboxes_car, probs=clean_scores_car, overlapThresh=self.car_nms_thr)
            clean_bboxes_car_nms = list(clean_bboxes_car[i] for i in pick_inds_car)
            clean_classes_pred_car_nms = list(clean_classes_pred_car[i] for i in pick_inds_car)
            clean_scores_car_nms = list(clean_scores_car[i] for i in pick_inds_car)

            clean_bboxes = clean_bboxes_pedestrian_nms + clean_bboxes_car_nms
            clean_classes_pred = clean_classes_pred_pedestrian_nms + clean_classes_pred_car_nms
            clean_scores = clean_scores_pedestrian_nms + clean_scores_car_nms

            # Tracker format:
            pedestrian_list = []
            car_list = []
            for bbox, score, label in zip(clean_bboxes, clean_scores, clean_classes_pred):
                # Need to redo the area filter due to the NMS
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                if area < 1024:
                    continue
                if label == 0:  # Pedestrian
                    pedestrian_list.append({"box2d": bbox, "score": score})
                elif label == 1:  # Car
                    car_list.append({"box2d": bbox, "score": score})
                else:
                    print("Irrelevant class detected: {}".format(label))
                    continue

            clean_detection = {"Car": car_list, "Pedestrian": pedestrian_list}

            return(clean_detection)

    def detect(self, img_inf):
        """Run predict on the image after preprocessing."""

        # Run inference
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(img_inf, axis=0))

        # Clean prediction output
        clean_detection = self.filter_prediction(boxes[0], scores[0], labels[0])

        return(clean_detection)

    def convert_to_signate(self, bbox, scores, classes_pred):
        """Convert model output into signate frame format:
            -input: bbox, scores, classes_pred
            -ouput: {'Car':[],'pedestrian':[]}
        """
        person_list = []
        car_list = []
        for bbox, score, cl in zip(bbox, scores, classes_pred):
            if score > 0:
                label = self.CLASSES[cl]

                if label == "Pedestrian":
                    person_list.append({"box2d":bbox})
                else:
                    car_list.append({"box2d":bbox})

        # add in the frame (if not empty)
        current_frame = {}
        if car_list:
            current_frame["Car"] = car_list
        if person_list:
            current_frame["Pedestrian"] = person_list

        return(current_frame)

    def display_on_frame(self, frame, pred_tracking):
        """Display all filtered bboxs and annotations on frame."""
        for cls, annot in pred_tracking.items():
            color = (255, 0, 0) if cls=="Pedestrian" else (0, 0, 255)
            for a in annot:
                xmin, ymin, xmax, ymax = list(map(int, a['box2d']))
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4)
                text = str(a['id'])
                cv2.putText(frame, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
