
from retinanet_wrapper import retinanet_inference
import signate_sub
from object_tracker import Tracker

import numpy as np
from PIL import Image
import cv2
import time

import nanocamera as nano

def main():

    # Init model - 640x360 resolution
    model = retinanet_inference("../models/small_resolution_resnet50_csv_11.h5")

    # Get camera
    camera = nano.Camera(camera_type=1, device_id=0, width=640, height=480, fps=30)
    print('USB Camera ready? - ', camera.isReady())

    # Prepare for submittion
    signate_output = signate_sub.signate_submission(model.classes_list)

    # Setup tracker
    tracker = Tracker((640, 480))

    frame_id = 0

    while camera.isReady():
        prev_time = time.time()
        frame = camera.read()
        print("frame raw: {}".format(frame.shape))

        # Resize frame
        inf_frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
        print("frame int: {}".format(inf_frame.shape))

        try:
            # Detection
            boxes, scores, classes_pred, pred_detection = model.detect(inf_frame)
            print("\nInference time: {} sec.".format(round((time.time()-prev_time), 3)))
            print("pred_detection: {}".format(pred_detection))

            # Tracking
            pred_tracking = tracker.assign_ids(pred_detection, frame)
            print("Tracking: {}".format(pred_tracking))

            # Post processing
            signate_output.display_on_frame(frame, pred_tracking)

            # Save frame for debugging
            if pred_detection:
                cv2.imwrite("../jetson_frame/" + str(frame_id) + ".png", frame)
                frame_id += 1

            print("Full run time: {} sec.".format(round((time.time()-prev_time), 3)))

        except Exception as e:
            print("Unable to process frame: {}".format(e))


if __name__ == "__main__":
    main()
