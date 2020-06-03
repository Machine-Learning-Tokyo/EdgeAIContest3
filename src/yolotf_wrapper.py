# yolov4-tf2 wrapper
import tensorflow as tf

##################
## Needed for my current setup
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow_addons as tfa
tfa.options.TF_ADDONS_PY_OPS = True
##################

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

import cv2
import matplotlib.pyplot as plt

# Default yolo classes:
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class yolov4_inference():
    def __init__(self, weight_path, threshold=0.5, classe_filter=['car', 'person']):
        # Model
        self.weight_path = weight_path
        self.threshold = threshold
        self.model = None
        self.classe_filter = classe_filter
        self.classes_list = CLASSES

        # Image setting
        self.height_in, self.width_in = (640, 960)
        self.height_out, self.width_out= (1936, 1216)

        # Load model
        self.load_model()
        self.model.summary()

    def load_model(self):
        print("Loading model...")
        self.model = YOLOv4(input_shape=(self.height_in, self.width_in, 3),
                anchors = YOLOV4_ANCHORS,
                num_classes = 80,
                training = False,
                yolo_max_boxes = 100,
                yolo_iou_threshold = self.threshold,
                yolo_score_threshold = self.threshold)

        try:
            self.model.load_weights(self.weight_path)
        except Exception as e:
            print("Unable to load weight file: {}".format(e))
        print("   ...Done!")

    def filter_prediction(self, boxes, scores, classes_pred):
        """Format prediction for tracking."""
        clean_bboxs = []
        clean_classes_pred = []
        clean_scores = []
        for bbox, score, cl in zip(boxes.tolist(), scores.tolist(), classes_pred.tolist()):
            if score > 0:
                label = CLASSES[cl]
                bbox = list(map(int, bbox))
                if label in self.classe_filter:
                    clean_bboxs.append(bbox)
                    clean_classes_pred.append(cl)
                    clean_scores.append(score)
        return([clean_bboxs, clean_scores, clean_classes_pred])

    def detect(self, cv2_image):
        """Run predict on the image after preprocessing."""
        # Convert image for network
        img_inf = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img_inf = tf.image.resize(img_inf, (self.height_in, self.width_in))
        img_inf = tf.expand_dims(img_inf, axis=0) / 255.0

        # Run inference
        bbox, scores, classes_pred, valid_detections = self.model.predict(img_inf)

        # Convert bbox into image position
        bbox = bbox[0]* [self.height_out, self.width_out, self.height_out, self.width_out]

        # Clean prediction output
        bbox, scores, classes_pred = self.filter_prediction(bbox, scores[0], classes_pred[0].astype(int))

        return([bbox, scores, classes_pred])

    def display_on_frame(self, frame, boxes, scores, classes_pred):
        """Display all filtered bboxs and annotations on frame."""
        for bbox, score, cl in zip(boxes, scores, classes_pred):
            if score > 0:
                label = CLASSES[cl]
                color = (255, 0, 0) if label=="person" else (0,0,255)

                # Bbox processing
                xmin, ymin, xmax, ymax = list(map(int, bbox))

                # Filter classes
                if label in self.classe_filter:
                    text = f'{CLASSES[cl]}: {score:0.2f}'
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4)
                    #cv2.putText(frame, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return(frame)