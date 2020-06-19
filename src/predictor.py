import glob
import os
import cv2
import pickle
from retinanet_wrapper import retinanet_inference
from object_tracker import Tracker


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model'):
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
        cls.model = retinanet_inference(
            model_path + "/resnet50_csv_01.h5.frozen")
        return True

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
        print("predict called")
        predictions = []
        cap = cv2.VideoCapture(input)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Init Tracker
        tracker = Tracker((w, h))

        fname = os.path.basename(input)
        i = 0
        while True:
            i = i + 1
            ret, frame = cap.read()
            print("inside loop ret = {}  i={}".format(ret, i))
            if not ret:
                break
            try:
                # Detection
                boxes, scores, classes_pred, pred_detection = cls.model.detect(frame)
                # Tracking
                pred_tracking = tracker.assign_ids(pred_detection, frame)
                #print("Tracking: {}".format(pred_tracking))
                predictions.append(pred_tracking)

            except Exception as e:
                print("Unable to process frame: {}".format(e))

            # if cls.model is not None:
            #     prediction = cls.model.predict(frame)
            # else:
            # prediction = {"Car": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}],
            #                 "Pedestrian": [{"id": 0, "box2d": [0, 0, frame.shape[1], frame.shape[0]]}]}

        cap.release()

        return {fname: predictions}
