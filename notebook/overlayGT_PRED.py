import json
import cv2
import sys
sys.path.insert(0, './keras-retinanet')
from keras_retinanet.utils.visualization import draw_box, draw_caption

def display_text(image, b1, text):
    w = 50
    h = 15
    line_height = 15
    i = 0
    cv2.rectangle(image, (b1[0], b1[1]), (b1[0] + w, b1[1] + h) , (0,0,0), -1)
    cv2.putText(image, text, (b1[0], b1[1] + h), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def process_video(current_file):
    with open("./train_annotations/{}.json".format(current_file)) as f:
        train_json = json.load(f)
    with open("./1_baseline/src/{}.preds.json".format(current_file)) as f:
        train_pred_json = json.load(f)

    cap = cv2.VideoCapture("./train_videos/{}.mp4".format(current_file))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps = {}".format(fps))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./train_videos/{}_overlayed_output.mp4".format(current_file), fourcc, fps, (int(w), int(h)))
    # out.set(cv2.CAP_PROP_FPS, fps)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    frame_count = 0
    # Read until video is completed
    while (cap.isOpened()):
        if frame_count%100 == 0:
            print("Frames processed {}".format(frame_count))
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # visualize ground truth
            if 'Pedestrian' in train_json['sequence'][frame_count].keys():
                pedestrians_list = train_json['sequence'][frame_count]['Pedestrian']
                for box in pedestrians_list:
                    b = box['box2d']
                    area = (b[2]-b[0])*(b[3]-b[1])
                    if (area >= 1024):
                        b = list(map(int, b))
                        draw_box(image, b, color=(0, 255, 0))
                    #draw_caption(image, b, str(box["id"]))
            if 'Car' in train_json['sequence'][frame_count].keys():
                car_list = train_json['sequence'][frame_count]['Car']
                for box in car_list:
                    b = box['box2d']
                    b = list(map(int, b))
                    area = (b[2]-b[0])*(b[3]-b[1])
                    if (area >= 1024):
                        draw_box(image, b, color=(0, 255, 0))
                    #draw_caption(image, b, str(box["id"]))
            
            # visualize predictions
            if 'Pedestrian' in train_pred_json['{}.mp4'.format(current_file)][frame_count].keys():
                pedestrians_list = train_pred_json['{}.mp4'.format(
                    current_file)][frame_count]['Pedestrian']
                for box in pedestrians_list:
                    b = box['box2d']
                    b = list(map(int, b))
                    area = (b[2]-b[0])*(b[3]-b[1])
                    if (area < 1024):
                        print("PRED : Pedestrian Small box = {} Area = {}".format(b, area))
                    draw_box(image, b, color=(255, 0, 0))
                    #display_text(image, b, str(box["score"]))
            if 'Car' in train_pred_json['{}.mp4'.format(current_file)][frame_count].keys():
                car_list = train_pred_json['{}.mp4'.format(
                    current_file)][frame_count]['Car']
                for box in car_list:
                    b = box['box2d']
                    b = list(map(int, b))
                    area = (b[2]-b[0])*(b[3]-b[1])
                    if (area < 1024):
                        print("PRED : Car Small box = {} Area = {}".format(b, area))
                    draw_box(image, b, color=(255, 0, 0))
                    #display_text(image, b, str(box["score"]))

            frame_count = frame_count + 1
            # write the resulting frame
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(image)
    
        # Break the loop
        else:
            break
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

    # write the file
    out.release()

target_videos = [
    "train_00",
    "train_01",
    "train_02",
    "train_12",
    "train_16",
    "train_22"
]

for current_file in target_videos:
    print("Overlaying {} ...... ".format(current_file))
    process_video(current_file)
