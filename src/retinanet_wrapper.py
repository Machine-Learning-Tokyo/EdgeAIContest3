# yolov4-tf2 wrapper
import tensorflow as tf

##################
## Needed for my current setup
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

##################

# import keras
import keras

import sys
sys.path.insert(0, '../src/')

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Default classes:
labels_to_names = {0: 'Pedestrian', 1: 'Car'}

class retinanet_inference():
    def __init__(self, weight_path, threshold=0.5, classe_filter=['Pedestrian','Car']):
        # Model
        self.weight_path = weight_path
        self.threshold = threshold
        self.model = None
        self.classe_filter = classe_filter
        self.classes_list = self.CLASSES = labels_to_names
        self.label_limit = 1

        # Image setting
        self.height_in, self.width_in = (640, 960)
        self.height_out, self.width_out= (1936, 1216)

        # Load model
        self.load_model()
        self.model.summary()

    def load_model(self):
        print("Loading model: {}".format(self.weight_path))
        try:
            self.model = models.load_model(self.weight_path, backbone_name='resnet50')
        except Exception as e:
            print("Unable to load weight file: {}".format(e))
        print("   ...Done!")

    def filter_prediction(self, boxes, scores, classes_pred):
        """Format prediction for tracking."""
        clean_bboxs = []
        clean_classes_pred = []
        clean_scores = []

        for bbox, score, cl in zip(boxes.tolist(), scores.tolist(), classes_pred.tolist()):
            if cl > self.label_limit:
                continue
            
            if score > self.threshold:
                label = self.CLASSES[cl]
                bbox = list(map(int, bbox))
                
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area <= 1024:
                    print("area = {}".format(area))
                    continue

                if label in self.classe_filter:
                    clean_bboxs.append(bbox)
                    clean_classes_pred.append(cl)
                    clean_scores.append(score)

        return([clean_bboxs, clean_scores, clean_classes_pred])

    def detect(self, cv2_image):
        """Run predict on the image after preprocessing."""
        # Convert image for network
        img_inf = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img_inf = preprocess_image(img_inf)
        img_inf, scale = resize_image(img_inf)

        # Run inference
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(img_inf, axis=0))

        # Convert bbox into image position
        boxes /= scale

        # Clean prediction output
        bbox, scores, classes_pred = self.filter_prediction(boxes[0], scores[0], labels[0])

        # Convert to Signate frame output here:
        signate_detection = self.convert_to_signate(bbox, scores, classes_pred)
        return([bbox, scores, classes_pred, signate_detection])

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

    def display_on_frame(self, frame, boxes, scores, classes_pred):
        """Display all filtered bboxs and annotations on frame."""
        for bbox, score, cl in zip(boxes, scores, classes_pred):
            if score > 0:
                label = self.CLASSES[cl]
                color = (255, 0, 0) if label=="Pedestrian" else (0,0,255)

                # Bbox processing
                xmin, ymin, xmax, ymax = list(map(int, bbox))

                # Filter classes
                if label in self.classe_filter:
                    text = f'{self.CLASSES[cl]}: {score:0.2f}'
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4)
                    #cv2.putText(frame, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return(frame)
