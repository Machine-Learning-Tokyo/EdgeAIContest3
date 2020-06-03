# Process pipeline draft
from yolotf_wrapper import yolov4_inference
import signate_sub

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse, time

def main():
    # Get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, help='Path to video', dest='video_input', default='test.mp4')
    parser.add_argument('-w', '--weights', type=str, help='Path to weight.h5', dest='weight_path', default='yolov4.h5')
    parser.add_argument('-m', '--model', type=str, help='Model version', dest='model_arch', default='yolov4')
    args = parser.parse_args()

    # Init model
    if args.model_arch == "yolov4":
        print("Detection backend running YOLO")
        model = yolov4_inference(args.weight_path)
    else:
        # Set retinanet_inference_wrapper here
        print("Detection backend running RETINANET")

    # Prepare for submittion
    signate_output = signate_sub.signate_submission(model.classes_list)

    # Load video
    cap = cv2.VideoCapture(args.video_input)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (int(w), int(h)))

    ret = True
    while ret:
        prev_time = time.time()
        ret, frame = cap.read()

        try:
            # Detection
            boxes, scores, classes_pred = model.detect(frame)
            print("\nInference time: {} ms.".format(round(1/(time.time()-prev_time), 3)))

            # Tracking
            ids = np.random.randint(100, size=len(boxes))
            print("Currently tracking: {} objects".format(len(ids)))

            # Generate Signate format
            signate_output.add_frame(boxes, classes_pred, scores, ids)

            # Display on frame
            model.display_on_frame(frame, boxes, scores, classes_pred)

            cv2.imshow('Demo', frame)
            out.write(frame)
            cv2.waitKey(3)

        except Exception as e:
            print("Unable to process frame: {}".format(e))

    # Save output
    print("Saving video output")
    cap.release()
    out.release()

    # Generate Submittion
    signate_output.write_submit()

if __name__ == "__main__":
    main()