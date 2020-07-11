# -*- coding: utf-8 -*-
# -----------------------------------------------------------
#  Main.py file to start the inference process.
#  This class intantiates a ScoringService and calls predict function of that instance.
#  The video file on which we want to perform inference (object detection and tracking) is passed as file path parameter to the predict function.
#  The results are returned in json format consisting of class predictions and assigned ids for each frame.
#  Multiple videos are passed by looping over the predict function and the results are aggregated into one json file.
#  Released under Apache License 2.0
#  Email: machine.learning.tokyo@gmail.com
# -----------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pdb
import json
import time
from predictor import ScoringService

exp_name = "flip_lr.aspect."

if ScoringService.get_model():
    start_time = time.time()
    target_videos = [
        "train_00",
        "train_01",
        "train_02",
        "train_12",
        "train_16",
        "train_22"
    ]
    # target_videos = ["train_{:02d}".format(i) for i in range(0,25)]

    combined_prediction_json = {}
    print("Processing videos = {}".format(target_videos))
    for train_file in target_videos:
        start_time_train = time.time()
        print("Train_file = {}".format(train_file))
        preds_json = ScoringService.predict(
            "/ext/signate_edge_ai/train_videos/{}.mp4".format(train_file))
        with open(exp_name + '{}.preds.json'.format(train_file), 'w+') as output_json_file:
            json.dump(preds_json, output_json_file)
        
        key = "{}.mp4".format(train_file)
        combined_prediction_json[key] = preds_json[key]
        end_time_train = time.time()
        print("[PERFORMANCE] ScoringService Video {} Total_Time = {}".format(
            train_file, end_time_train - start_time_train))
        print("-----------------------------")

    print("Outputing combined predictions ")
    with open(exp_name + 'prediction.json', 'w+') as output_combined_json_file:
        json.dump(combined_prediction_json, output_combined_json_file)
    end_time = time.time()
    print("[PERFORMANCE] ScoringService All_Videos {} Total_Time = {}".format(
        train_file, end_time - start_time))
# preds_json = ss.predict("/ext/signate_edge_ai/train_videos/train_01.mp4")
