# Simple submission helper
import json
import numpy as np
import os
import cv2
import sys

CLASSES = [ 'Pedestrian', 'Car']

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
    def __init__(self, classe_list, file_name="prediction.json"):
        self.ouput_file_name = file_name
        self.sequences = []
        self.videos = []
        self.out_json = {}

        self.filter = ["Pedestrian", "Car"]

    def add_frame_old(self, bbox, classes_pred, scores, ids):
        """Add a new frame to submission sequence."""
        person_list = []
        car_list = []
        for bbox, score, cl, _id in zip(bbox, scores, classes_pred, ids):
            if score > 0:
                label = CLASSES[cl]

                if label == "Pedestrian":
                    person_list.append({"id": _id, "box2d":bbox})
                elif label == "Car":
                     car_list.append({"id": _id, "box2d":bbox})
                else:
                    print("Wrong label in add_frame_old")

        # add in the frame (if not empty)
        current_frame = {}
        if person_list:
            current_frame["Pedestrian"] = person_list
        if car_list:
            current_frame["Car"] = car_list

        self.sequences.append([current_frame])

    def add_frame(self, pred_tracking):
        """Simply add the prediction to the video entrance."""
        self.sequences.append(pred_tracking)

    def write_video(self, video_name):
        """Add the video and frame to the output file"""
        self.out_json[str(video_name)]=self.sequences

    def write_submit(self):
        """Write final json file."""
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
