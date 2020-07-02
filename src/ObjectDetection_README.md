# Pre-processing
## Folder Structure
    .
    ├── generate_retinanet_train_annotation.py   # training data generator
    ├── generate_retinanet_val_annotation.py     # validation data generator
    ├── ObjectDetection_README.md                # Readme file
    ├── convert_video_to_image.sh                # quick script to extract individual frames as .png images from videos
    ├── train_annotations                        # train_annotations folder provided by signate (symlinked or copy)
    └── train_videos                             # train_videos folder provided by signate (symlinked or copy)

## Data format
We have extracted the frames from the video and saved the still images as PNG format using ffmpeg tool. Please install this tool using apt package manager. 

Run the "convert_video_to_image.sh" , it will consume the videos present in train_videos folder and produce train_{} folder for each training video, containing all the images.
```
    apt update
    apt install ffmpeg
    ffmpeg -version

    bash convert_video_to_image.sh
```

## Training data creation
- There are 10 classes in train set, for training we have used 5 of them: 
    - at first, we have trained the object detection model using all of these class annotations
    - but then, we thought it could be better to train on only Pedestrian and Car. This model's performance was acceptable
    - but then, we have seen that our model sometimes detecting the Trucks, Bus as the Car (confusion among vehicle classes)
    - we trained another model using the following class (5 classes) annotations: Pedestrian, Car, Truck, Bus, and Svehicle
- we have splitted the videos into train and validation set:
    - validation set: train_00, train_01, train_01
    - train set: train_02 ~ train_24
- the corresponding training data is generated using the script: [generate_retinanet_train_annotation.py](generate_retinanet_train_annotation.py)
    - we have seen some outliers (odd cases) where the annotation pixel value exceeds the image dimensions. That's why we substracted 3 from annotation values (to make it valid annotation), and this would not hurt the training at all, i.e. changing the bounding box annotation does not change anything at all.
    - excluded thin, short objects (if any). excluded the object with the width/height of smaller than 5 pixels.
    - the training data format is the same with keras-retinanet's CSV data format:
        - ```image_fpath,x1,y1,x2,y2,class```
    - this script generates a train data file:
            - ```retinanet_annotations.csv.train.all_frames.all_objects.5_classes```
- the corresponding training data is generated using the script: [generate_retinanet_val_annotation.py](generate_retinanet_val_annotation.py)
    - we have seen some outliers (odd cases) where the annotation pixel value exceeds the image dimensions. That's why we did not include those annotations
    - excluded thin, short objects (if any). excluded the object with the width/height of smaller than 5 pixels.
    - the training data format is the same with keras-retinanet's CSV data format:
        - ```image_fpath,x1,y1,x2,y2,class```
    - this script generates a validation data file:
            - ```retinanet_annotations.csv.val.all_frames.all_objects.5_classes```
- construct the `class_id_map.txt.5classes` file:
    ```
        Pedestrian,0
        Car,1
        Truck,2
        Bus,3
        Svehicle,4
    ```

# Model training
- For object detection, we have used keras-retinanet library: https://github.com/fizyr/keras-retinanet
    Here is a sample manual
    ```
        git clone https://github.com/fizyr/keras-retinanet.git
        cd keras-retinanet; pwd
        pip install . --user
        python setup.py build_ext --inplace
    ```
    - Installation: after cloning the repo to the machine, run `pip install . --user` 
    - We have modified the augmentation parameters in the `keras_retinanet/bin/train.py`. here is the augmentation we have used during the training:
    ```
      transform_generator = random_transform_generator(
        min_rotation=-0.2,
        max_rotation=0.2,
        min_translation=(-0.2, -0.2),
        max_translation=(0.2, 0.2),
        min_shear=-0.2,
        max_shear=0.2,
        min_scaling=(0.7, 0.7),
        max_scaling=(1.4, 1.4),
        flip_x_chance=0.5,
        flip_y_chance=0,
        )
    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.8, 1.2),
        brightness_range=(-.3, .3),
        hue_range=(-0.1, 0.1),
        saturation_range=(0.9, 1.1)
    )
    ```

    - Training: run the `keras_retinanet/bin/train.py` file with some arguments. The backbone pretrained model (ImageNet weights) will be downloaded automatically. 
        ```
        python keras-retinanet/keras_retinanet/bin/train.py --steps 13200 --snapshot-path ./all_obj.5classes.resnet101 --random-transform --no-resize --lr 1e-5 --epochs 100 --backbone resnet101 csv retinanet_annotations.csv.train.all_frames.all_objects.5_classes class_id_map.txt.5classes --val-annotations retinanet_annotations.csv.val.all_frames.all_objects.5_classes
      ```
        - steps: the total number of images in train set is 13,200: `22 (videos) * 600 (total number of frames in a video) = 13200`
        - snapshot-path: where to save the model snapshots
        - random-transform: to perform the pre-defined augmentations
        - no-resize: do not resize the original images, i.e. input image size to the model is `1936 x 1216`
        - lr: learning rate
        - epochs: total number of epochs for training
        - backbone: the CNN backbone used in retinanet framework. we have used resnet101
        - csv: data format for training the retinanet model. train set and id_class_map files provided after this argument
        - val-annotations: validation set to calculate the mAP after each epoch
    - Freeze the model: run `keras_retinanet/bin/convert_model.py resnet101_csv_15.h5 resnet101_csv_15.h5.frozen`
        - the freezing step decreases the model size from `635M` to `213M` since freezing step does not store the gradients
     
