# Simple submission helper
import json
import numpy as np
import os
import cv2

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

class NpEncoder(json.JSONEncoder):
    # Issue when dumping the json: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class signate_submission():
    def __init__(self, classe_list, file_name="signate_output.json"):
        self.ouput_file_name = file_name
        self.sequences = []
        self.out_json = []

        self.filter = ["person", "car"]

    def add_frame_old(self, bbox, classes_pred, scores, ids):
        """Add a new frame to submission sequence."""
        person_list = []
        car_list = []
        for bbox, score, cl, _id in zip(bbox, scores, classes_pred, ids):
            if score > 0:
                label = CLASSES[cl]

                if label == "person":
                    person_list.append({"id": _id, "box2d":bbox})

                else:
                    car_list.append({"id": _id, "box2d":bbox})

        # add in the frame (if not empty)
        current_frame = {}
        if person_list:
            current_frame["Pedestrian"] = person_list
        if car_list:
            current_frame["Car"] = car_list

        self.sequences.append([current_frame])

    def add_frame(self, pred_tracking):
        """Simply add the prediction to the output file."""
        self.sequences.append([pred_tracking])

    def write_submit(self):
        """Write final json file."""
        # Fill sequence with all frame
        self.out_json.append({"Sequence":self.sequences})

        # Write file on local
        with open(self.ouput_file_name, 'w+') as output_json_file:
            json.dump(self.out_json, output_json_file, cls=NpEncoder)

        output_json_file.close()
        print("Submission file generated: {}".format(self.ouput_file_name))

    def display_on_frame(self, frame, pred_tracking):
        """Display all filtered bboxs and annotations on frame."""
        for cls, annot in pred_tracking.items():
            color = (255, 0, 0) if cls=="Pedestrian" else (0, 0, 255)
            for a in annot:
                xmin, ymin, xmax, ymax = list(map(int, a['box2d']))
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 4)
                text = "id: " + str({a['id']})
                cv2.putText(frame, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
