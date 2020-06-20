import os
import cv2
import numpy as np
from keras_retinanet import models
from object_tracker import Tracker
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import copy
import time
import pdb


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model/resnet50_csv_06.h5.frozen'):
        """Get model method
 
        Args:
            model_path (str): Path to the trained model directory.
 
        Returns:
            bool: The return value. True for success, False otherwise.
        
        Note:
            - You cannot connect to external network during the prediction,
              so do not include such process as using urllib.request.
 
        """
        print("get_model called")
        try:
            cls.threshold_pedestrian = 0.5
            cls.threshold_car = 0.5
            cls.expansion = 0
            cls.scales = [0.2, 0.4]
            cls.model = models.load_model(
                '../model/resnet50_csv_06.h5.frozen', backbone_name='resnet50')
            return True
        except Exception as e:
            print("Failed to load model {}".format(e))
            return False

    @classmethod
    def non_max_suppression_fast(cls, boxes, overlapThresh=0.4):
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return pick

    @classmethod
    def model_inference(cls, frame):
        try:
            # detection
            wc = cls.w // 2
            hc = cls.h // 2

            img_inf0 = preprocess_image(frame)

            offset_x1_1 = int(wc - int(cls.w * cls.scales[0]))
            offset_y1_1 = int(hc - int(cls.h * cls.scales[0]))
            offset_x2_1 = int(wc + int(cls.w * cls.scales[0]))
            offset_y2_1 = int(hc + int(cls.h * cls.scales[0]))
            img_inf1 = img_inf0[offset_y1_1:offset_y2_1, offset_x1_1:offset_x2_1]

            offset_x1_2 = int(wc - int(cls.w * cls.scales[0]))
            offset_y1_2 = int(hc - int(cls.h * cls.scales[0]))
            offset_x2_2 = int(wc + int(cls.w * cls.scales[0]))
            offset_y2_2 = int(hc + int(cls.h * cls.scales[0]))
            img_inf2 = img_inf0[offset_y1_2:offset_y2_2, offset_x1_2:offset_x2_2]

            img_inf0, scale0 = resize_image(img_inf0)
            img_inf1, scale1 = resize_image(img_inf1)
            img_inf2, scale2 = resize_image(img_inf2)

            # boxes, scores, labels = cls.model.predict_on_batch(np.expand_dims(img_inf_batch, axis=0))
            boxes, scores, labels = cls.model.predict_on_batch(np.array([img_inf0, img_inf1, img_inf2]))

            boxes_0 = boxes[0] / scale0
            boxes_1 = boxes[1] / scale1
            boxes_2 = boxes[2] / scale2

            clean_bboxes_, clean_classes_pred_, clean_scores_ = [], [], []
            for bbox_, score_, label_ in zip(boxes_0, scores[0], labels[0]):
                if (label_ == 0 and score_ >= cls.threshold_pedestrian) or (label_ == 1 and score_ >= cls.threshold_car):

                    [x1, y1, x2, y2] = bbox_
                    width = x2 - x1
                    height = y2 - y1
                    if width * height < 4000:
                        x1_ = max(0, x1 - width * cls.expansion)
                        y1_ = max(0, y1 - height * cls.expansion)
                        x2_ = min(cls.w, x2 + width * cls.expansion)
                        y2_ = min(cls.h, y2 + height * cls.expansion)
                        bbox = [int(x1_), int(y1_), int(x2_), int(y2_)]
                    else:
                        bbox = list(map(int, bbox_))

                    clean_bboxes_.append(bbox)
                    clean_classes_pred_.append(label_)
                    clean_scores_.append(score_)

            for bbox_, score_, label_ in zip(boxes_1, scores[1], labels[1]):
                if (label_ == 0 and score_ >= cls.threshold_pedestrian) or (label_ == 1 and score_ >= cls.threshold_car):
                    [x1, y1, x2, y2] = bbox_
                    x1 += offset_x1_1
                    y1 += offset_y1_1
                    x2 += offset_x1_1
                    y2 += offset_y1_1
                    width = x2 - x1
                    height = y2 - y1
                    if width * height <= 1024:
                        continue

                    if width * height < 4000:
                        x1_ = max(0, x1 - width * cls.expansion)
                        y1_ = max(0, y1 - height * cls.expansion)
                        x2_ = min(cls.w, x2 + width * cls.expansion)
                        y2_ = min(cls.h, y2 + height * cls.expansion)
                        bbox = [int(x1_), int(y1_), int(x2_), int(y2_)]
                    else:
                        bbox = list(map(int, bbox_))
                    clean_bboxes_.append(bbox)
                    clean_classes_pred_.append(label_)
                    clean_scores_.append(score_)

            for bbox_, score_, label_ in zip(boxes_2, scores[2], labels[2]):
                if (label_ == 0 and score_ >= cls.threshold_pedestrian) or (label_ == 1 and score_ >= cls.threshold_car):
                    [x1, y1, x2, y2] = bbox_
                    x1 += offset_x1_2
                    y1 += offset_y1_2
                    x2 += offset_x1_2
                    y2 += offset_y1_2
                    width = x2 - x1
                    height = y2 - y1
                    if width * height <= 1024:
                        continue
                    if width * height < 4000:
                        x1_ = max(0, x1 - width * cls.expansion)
                        y1_ = max(0, y1 - height * cls.expansion)
                        x2_ = min(cls.w, x2 + width * cls.expansion)
                        y2_ = min(cls.h, y2 + height * cls.expansion)
                        bbox = [int(x1_), int(y1_), int(x2_), int(y2_)]
                    else:
                        bbox = list(map(int, bbox_))

                    clean_bboxes_.append(bbox)
                    clean_classes_pred_.append(label_)
                    clean_scores_.append(score_)

            pick_inds = cls.non_max_suppression_fast(np.array(clean_bboxes_))
            clean_bboxes = list(clean_bboxes_[i] for i in pick_inds)
            clean_classes_pred = list(clean_classes_pred_[i] for i in pick_inds)
            clean_scores = list(clean_scores_[i] for i in pick_inds)
            pedestrian_list = []
            car_list = []
            for bbox, score, label in zip(clean_bboxes, clean_scores, clean_classes_pred):
                if label == 0:  # Pedestrian
                    pedestrian_list.append({"box2d": bbox})
                else:  # Car
                    car_list.append({"box2d": bbox})

            current_frame = {"Car": car_list, "Pedestrian": pedestrian_list}

            pred_tracking = cls.tracker.assign_ids(current_frame, frame)

            return pred_tracking
        except:
            return None

    @classmethod
    def predict(cls, input):
        """Predict method
 
        Args:
            input (str): path to the video file you want to make inference from
 
        Returns:
            dict: Inference for the given input.
                format:
                    - filename []:
                        - category_1 []:
                            - id: int
                            - box2d: [left, top, right, bottom]
                        ...
        Notes:
            - The categories for testing are "Car" and "Pedestrian".
              Do not include other categories in the prediction you will make.
            - If you do not want to make any prediction in some frames,
              just write "prediction = {}" in the prediction of the frame in the sequence(in line 65 or line 67).
        """
        print("Predicting file {}".format(input))

        predictions = []
        cap = cv2.VideoCapture(input)
        cls.w, cls.h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT)
        cls.tracker = Tracker((cls.w, cls.h))
        # prev_prediction = {"Car": [{"id": 0, "box2d": [0, 0, w, h]}], "Pedestrian": [{"id": 0, "box2d": [0, 0, w, h]}]}
        prev_prediction = {}
        fname = os.path.basename(input)
        ii = 0
        while True:
            start_time = time.time()
            ii += 1
            if (ii % 100 == 0) :
                print("Frames processed : {}".format(ii))
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if cls.model is not None:
                    prediction = cls.model_inference(frame)
                    if prediction is None:
                        prediction = prev_prediction
                    predictions.append(prediction)
                    prev_prediction = copy.copy(prediction)
                else:
                    # prediction = {"Car": [{"id": 0, "box2d": [0, 0, w, h]}], "Pedestrian": [{"id": 0, "box2d": [0, 0, w, h]}]}
                    prediction = {}
                    predictions.append(prediction)
            except:
                predictions.append(prev_prediction)
        cap.release()
        end_time = time.time()
        print("[PERFORMANCE] Video {} Frame {} Total_Time     = {}".format(
            fname, ii, end_time - start_time))
        print("-----------------------------")
        return {fname: predictions}
