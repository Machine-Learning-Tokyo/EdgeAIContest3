
from retinanet_wrapper import retinanet_inference
from object_tracker import Tracker

import numpy as np
from PIL import Image
import cv2
import time, argparse, os

import nanocamera as nano

def check_directory(saving_directory):
    # Check if saving directory exist, create if not.
    if not os.path.isdir(saving_directory):
        os.makedirs(saving_directory)
        return(True)
    else:
        return(False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help="Model file (.tflite)", dest='model_file', required=True)
    parser.add_argument('--csi', help='CSI camera type', dest='cam_csi', action='store_const', const=True)
    parser.add_argument('--OUT', help='gstreamer ouput', dest='out', action='store_const', const=True)
    args = parser.parse_args()

    # Init model - 640x360 resolution
    model = retinanet_inference(args.model_file)

    # Get camera
    if args.cam_csi:
        print("Getting CSI camera...")
        # Add Naveen code here.
        frame_w = 640
        frame_h = 360

    else:
        print("Grabing USB camera...")
        frame_w = 640
        frame_h = 480
        camera = nano.Camera(camera_type=1, device_id=0, width=frame_w, height=frame_h, fps=30)
        print('USB Camera ready? - ', camera.isReady())

    if args.out:
        # Add args for variable IP and port
        out_pipeline = "appsrc ! videoconvert ! video/x-raw,format=NV12 ! jpegenc ! rtpjpegpay !  queue ! udpsink host=192.168.3.3 port=1234 sync=false"
        out = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, 28, (frame_w, frame_h))

    # Setup tracker
    tracker = Tracker((frame_w, frame_h))
    frame_id = 0

    # Generate directory for saving detection frame:
    saving_path = "jetson_frame/"
    check_directory(saving_path)

    while camera.isReady():
        prev_time = time.time()
        inf_frame = camera.read()

        try:
            # Detection
            pred_detection = model.detect(inf_frame)
            print("\nInference time: {} sec.".format(round((time.time()-prev_time), 3)))
            print("pred_detection: {}".format(pred_detection))

            # Tracking
            if pred_detection:
                pred_tracking = tracker.assign_ids(pred_detection, inf_frame)
                print("Tracking: {}".format(pred_tracking))

            # Save frame for debugging
            if pred_detection:
                model.display_on_frame(inf_frame, pred_tracking)
                cv2.imwrite(saving_path + str(frame_id) + ".png", inf_frame)
                frame_id += 1

            # Stream Ouput
            if args.out:
                out.write(inf_frame)

            print("Full Loop time: {} sec.".format(round((time.time()-prev_time), 3)))

        except Exception as e:
            print("Unable to process frame: {}".format(e))

if __name__ == "__main__":
    main()
