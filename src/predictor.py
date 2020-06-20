import os
import cv2
import numpy as np
from keras_retinanet import models
from object_tracker import Tracker
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import copy
import time

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model/resnet50_csv_01.h5.frozen'):
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
            #cls.tracker = Tracker((1936, 1216))
            cls.model = models.load_model(
                '../model/resnet50_csv_01.h5.frozen', backbone_name='resnet50')
            return True
        except:
            return False

    @classmethod
    def model_inference(cls, frame):
        print("model_inference called")
        try:
            # detection
            start_time_detection = time.time()
            img_inf = preprocess_image(frame)
            img_inf, scale = resize_image(img_inf)
            boxes, scores, labels = cls.model.predict_on_batch(
                np.expand_dims(img_inf, axis=0))
            boxes /= scale

            clean_bboxes, clean_classes_pred, clean_scores = [], [], []
            for bbox, score, label in zip(boxes[0], scores[0], labels[0]):
                if (label == 0 and score >= cls.threshold_pedestrian) or (label == 1 and score >= cls.threshold_car):
                    bbox = list(map(int, bbox))

                    clean_bboxes.append(bbox)
                    clean_classes_pred.append(label)
                    clean_scores.append(score)

            pedestrian_list = []
            car_list = []
            for bbox, score, label in zip(clean_bboxes, clean_scores, clean_classes_pred):
                if label == 0:  # Pedestrian
                    pedestrian_list.append({"box2d": bbox})
                else:  # Car
                    car_list.append({"box2d": bbox})

            current_frame = {"Car": car_list, "Pedestrian": pedestrian_list}
            end_time_detection = time.time()


            print("[PERFORMANCE] Detection_Time = {}".format(end_time_detection - start_time_detection))
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
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT)
        cls.tracker = Tracker((w, h))
        prev_prediction = {"Car": [{"id": 0, "box2d": [0, 0, w, h]}], "Pedestrian": [
            {"id": 0, "box2d": [0, 0, w, h]}]}
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
                        prediction = {}
                    predictions.append(prediction)
                    prev_prediction = copy.copy(prediction)
                else:
                    prediction = {"Car": [{"id": 0, "box2d": [0, 0, w, h]}], "Pedestrian": [
                        {"id": 0, "box2d": [0, 0, w, h]}]}
                    predictions.append(prediction)
            except:
                predictions.append(prev_prediction)
        cap.release()
        if len(predictions) > ii:
            predictions = predictions[:ii]
        end_time = time.time()
        print("[PERFORMANCE] Video {} Frame {} Total_Time     = {}".format(
            fname, ii, end_time - start_time))
        print("-----------------------------")
        #else:
        #    diff = ii - len(predictions)
        #    for i in range(diff):
        #        predictions.append({})
        return {fname: predictions}
