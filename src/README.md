# Source Explained

**TODO:**

- Tracking integration
- Uniform prediction
- Retinanet wrapper
- Check submission format

## Content
Compossed by 1 main process: ```process_video.py```, that takes care of loading the video, setting up the **detection_module**, the **submission_helper**, and **tracking_module**.

- Detection module: is a wrapper of tensorflow implementation, that generate formated prediction.

- Tracking module: generate tracking based on previous-current frame transformation and detection output.

- Submission helper: generate a submission **.json** file.