import os
import pdb
import json

from predictor import ScoringService

if ScoringService.get_model():
    preds_json = ScoringService.predict("/ext/signate_edge_ai/train_videos/train_01.mp4")
    with open('train_01.preds.json', 'w+') as output_json_file:
        json.dump(preds_json, output_json_file)

# preds_json = ss.predict("/ext/signate_edge_ai/train_videos/train_01.mp4")
