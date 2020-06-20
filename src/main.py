import os
import pdb
import json
import time
from predictor import ScoringService

if ScoringService.get_model():
    videos_path = "/Users/yoovraj.shinde/work/signate/train_videos"
    start_time = time.time()
    for i in range(0, 25):
        start_time_train = time.time()
        train_file = "train_{:02d}".format(i)
        preds_json = ScoringService.predict(
            "/Users/yoovraj.shinde/work/signate/train_videos/{}.mp4".format(train_file))
        with open('{}.preds.json'.format(train_file), 'w+') as output_json_file:
            json.dump(preds_json, output_json_file)
        end_time_train = time.time()
        print("[PERFORMANCE] ScoringService Video {} Total_Time = {}".format(
            train_file, end_time_train - start_time_train))
        print("-----------------------------")

    end_time = time.time()
    print("[PERFORMANCE] ScoringService All_Videos {} Total_Time = {}".format(
        i, end_time - start_time))
# preds_json = ss.predict("/ext/signate_edge_ai/train_videos/train_01.mp4")
