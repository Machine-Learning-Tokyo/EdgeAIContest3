# Default classes:
labels_to_names_def = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                       39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
labels_to_names = {0: 'Pedestrian', 1: 'Car'}


class dummy_inference():
    def __init__(self, weight_path, threshold=0.5, classe_filter=['Car', 'Pedestrian']):
        # Model
        self.weight_path = weight_path
        self.threshold = threshold
        self.model = None
        self.classe_filter = classe_filter
        self.classes_list = self.CLASSES = labels_to_names
        self.label_limit = 1

        # Image setting
        self.height_in, self.width_in = (640, 960)
        self.height_out, self.width_out = (1936, 1216)

        # Load model
        self.load_model()
        self.model.summary()

    def load_model(self):
        print("Loading model: {}".format(self.weight_path))

    def filter_prediction(self, boxes, scores, classes_pred):
        """Format prediction for tracking."""
        clean_bboxs = []
        clean_classes_pred = []
        clean_scores = []
        return([clean_bboxs, clean_scores, clean_classes_pred])

    def detect(self, cv2_image):
        """Run predict on the image after preprocessing."""
        # Convert image for network
        # Clean prediction output
        bbox, scores, classes_pred = self.filter_prediction(
            boxes[0], scores[0], labels[0])

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
                    person_list.append({"box2d": bbox})

                else:
                    car_list.append({"box2d": bbox})

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
                color = (255, 0, 0) if label == "Pedestrian" else (0, 0, 255)

                # Bbox processing
                xmin, ymin, xmax, ymax = list(map(int, bbox))

                # Filter classes
                if label in self.classe_filter:
                    text = f'{self.CLASSES[cl]}: {score:0.2f}'
                    cv2.rectangle(frame, (int(xmin), int(ymin)),
                                  (int(xmax), int(ymax)), color, 4)
                    #cv2.putText(frame, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return(frame)
