# Process pipeline draft
# run with RETINANET: python process_video.py -v ../data/train_videos/train_00.mp4 -m retinanet -w ../model/resnet50_sub1.h5.frozen
# or with YOLOV4 python process_video.py -v ../data/train_videos/train_00.mp4

from retinanet_wrapper import retinanet_inference
import signate_sub
from object_tracker import Tracker

import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt
import argparse, time


def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, help='Path to video', dest='video_input', default='test.mp4')
    parser.add_argument('-w', '--weights', type=str, help='Path to weight.h5', dest='weight_path', default='yolov4.h5')
    parser.add_argument('-m', '--model', type=str, help='Model version', dest='model_arch', default='yolov4')
    parser.add_argument('-d', '--display', help='display frame', dest='display', action='store_const', const=True)
    parser.add_argument('-o', '--output', help='video output', dest='video_out', action='store_const', const=True)
    parser.set_defaults(display=False)
    parser.set_defaults(video_out=False)
    args = parser.parse_args()

    # Init model
    if args.model_arch == "yolov4":
        print("Detection backend running YOLO")
        model = yolov4_inference(args.weight_path)
    else:
        # Set retinanet_inference_wrapper here
        print("Detection backend running RETINANET")
        model = retinanet_inference(args.weight_path)

    # Prepare for submittion
    signate_output = signate_sub.signate_submission(model.classes_list)

    # Load video
    cap = cv2.VideoCapture(args.video_input)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if args.video_out:
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (int(w), int(h)))

    # Init Tracker
    tracker = Tracker((w, h))

    ret = True
    while ret:
        prev_time = time.time()
        ret, frame = cap.read()

        try:
            # Detection
            boxes, scores, classes_pred, pred_detection = model.detect(frame)
            # print("\nInference time: {} ms.".format(round(1/(time.time()-prev_time), 3)))
            # print("pred_detection: {}".format(pred_detection))

            # Tracking
            pred_tracking = tracker.assign_ids(pred_detection, frame)
            print("Tracking: {}".format(pred_tracking))

            # Generate Signate format
            signate_output.add_frame(pred_tracking)

            # Display on frame
            signate_output.display_on_frame(frame, pred_tracking)
            if args.display:
                cv2.imshow('Demo', frame)
                cv2.waitKey(3)

            # Write output video
            if args.video_out:
                out.write(frame)

        except Exception as e:
            print("Unable to process frame: {}".format(e))

    # Write video prediction to output file
    signate_output.write_video("train_00.mp4")

    # Save output
    print("Saving video output")
    cap.release()
    if args.video_out:
        out.release()

    # Generate Submittion
    signate_output.write_submit()


if __name__ == "__main__":
    main()