import glob
import os
import cv2
import pickle
import pdb
import numpy as np
from keras_retinanet import models
from object_tracker import Tracker
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


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
        cls.threshold = 0.5
        cls.tracker = Tracker((1936, 1216))
        cls.model = models.load_model(model_path, backbone_name='resnet50')
        return True

    @classmethod
    def model_inference(cls, frame):
        img_inf = preprocess_image(frame)
        img_inf, scale = resize_image(img_inf)
        boxes, scores, labels = cls.model.predict_on_batch(np.expand_dims(img_inf, axis=0))
        boxes /= scale

        clean_bboxes, clean_classes_pred, clean_scores = [], [], []
        for bbox, score, label in zip(boxes[0], scores[0], labels[0]):
            if score > cls.threshold:
                bbox = list(map(int, bbox))
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area <= 1024:
                    print("area = {}".format(area))
                    continue

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

        pred_tracking = cls.tracker.assign_ids(current_frame, frame)

        return pred_tracking

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

        predictions = []
        cap = cv2.VideoCapture(input)
        fname = os.path.basename(input)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cls.model is not None:
                prediction = cls.model_inference(frame)
            else:
                prediction = {"Car": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}],
                              "Pedestrian": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}]}
            predictions.append(prediction)
        cap.release()
        
        return {fname: predictions}
