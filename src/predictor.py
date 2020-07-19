# -*- coding: utf-8 -*-
# -----------------------------------------------------------
# This file is the sample submission file according to Signate runtime submission format.
#  Details of the template can be checked on https://signate.jp/features/runtime/detail
#  Released under Apache License 2.0
#  Email: machine.learning.tokyo@gmail.com
# -----------------------------------------------------------
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


import os
import cv2
import numpy as np
from keras_retinanet import models
from object_tracker import Tracker
from keras_retinanet.utils.image import read_image_bgr, adjust_brightness
import copy
from keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
import time
import pdb
from collections import defaultdict




class ScoringService(object):
    @classmethod
    def get_model(cls, model_path='../model/resnet50_csv_06.h5.frozen'):
        """ Method to load a model by specified file path. 
            Returns False if model cannot be loaded properly."""
        
        print("get_model called")
        try:
            cls.min_no_of_frames = 2  # 2 seems more reasonable than 4 !!!
            cls.left_crop = False
            cls.right_crop = False
            cls.flip_lr = True
            cls.bright_frame = False
            cls.dark_frame = False
            cls.pedestrian_nms_thr = 0.4
            cls.car_nms_thr = 0.35
            cls.conf_score_bias = 0.1
            cls.reassign_id_pedestrian = False
            cls.dummy_id_range = range(5000, 5555)

            cls.threshold_pedestrian = 0.5  # DO NOT CHANGE
            cls.threshold_car = 0.5  # DO NOT CHANGE
            cls.expansion = 0  # DO NOT USE
            cls.scales = [0.2]  #  DO NOT CHANGE
            cls.apply_heuristic_post_processing = True  # ALWAYS USE THIS HEURISTIC !!!
            cls.apply_adaptive_pedestrian_nms = False

            cls.model = models.load_model('../model/resnet101_csv_15.5classes.all_bboxes.h5.frozen', backbone_name='resnet101')  # 0.6358
            # batch_size, 1216, 1936, 3
            # _, _, _ = cls.model.predict_on_batch(np.zeros((2, 1216, 1936, 3)))

            cls.car_classification_model = load_model('../model/car_model_074.h5', compile=False)
            cls.pedestrian_classification_model = load_model('../model/pedestrian_model_epoch_3_068.h5', compile=False)
            return True
        except Exception as e:
            print("Failed to load model {}".format(e))
            return False

    @classmethod
    def non_max_suppression_with_scores(cls, boxes, probs=None, overlapThresh=0.5):
        """ Eliminates the bounding boxes according to the prediction scores according to iou threshold) """
        boxes = np.array(boxes)
        probs = np.array(probs)
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        idxs = y2

        if probs is not None:
            idxs = probs

        idxs = np.argsort(idxs)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            xx1_union = np.minimum(x1[i], x1[idxs[:last]])
            yy1_union = np.minimum(y1[i], y1[idxs[:last]])
            xx2_union = np.maximum(x2[i], x2[idxs[:last]])
            yy2_union = np.maximum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            w_union = np.maximum(0, xx2_union - xx1_union + 1)
            h_union = np.maximum(0, yy2_union - yy1_union + 1)

            # overlap = (w * h) / area[idxs[:last]]
            iou = (w * h) / (w_union * h_union)

            # idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > overlapThresh)[0])))

        return pick

    @classmethod
    def compute_resize_scale(cls, image_shape, min_side=1216, max_side=1936):
        """ Compute the resize scale value to resize the image """
        (rows, cols, _) = image_shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        return scale

    @classmethod
    def resize_image(cls, img, min_side=1216, max_side=1936):
        """ Resizes the input image to the given input shape """
        scale = cls.compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        return img, scale

    @classmethod
    def preprocess_image(cls, x):
        """ Preprocess the image, i.e. normalize the color (RGB) channels (x-=[103.939, 116.779, 123.68]) """
        x = x.astype(np.float32)
        x -= [103.939, 116.779, 123.68]
        return x

    @classmethod
    def draw_bboxes(cls, bboxes, image):
        """ This is not used in test set, this function utilized during the experimentation - draw the bounding boxes in given image """
        for bbox in bboxes:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 1)
        return image

    @classmethod
    def apply_local_nms(cls, clean_bboxes, clean_classes_pred, clean_scores):
        """ Applies nms on given bounding boxes for merging the different predictions, 
        This function calls non_max_suppression_with_scores function inside (with overlap of 0.8. 
        This threshold is high not to lose much bounding boxes because we apply the global nms afterwards).
        """
        pick_inds = cls.non_max_suppression_with_scores(clean_bboxes, probs=clean_scores, overlapThresh=0.8)
        clean_bboxes = list(clean_bboxes[i] for i in pick_inds)
        clean_classes_pred = list(clean_classes_pred[i] for i in pick_inds)
        clean_scores = list(clean_scores[i] for i in pick_inds)

        return clean_bboxes, clean_classes_pred, clean_scores

    @classmethod
    def apply_heuristics(cls, clean_bboxes_, clean_classes_pred_, clean_scores_, offset_y1_1, offset_y2_1):
        """ Here we apply some heuristics after all detection processes, 
        i.e. remove the detected bounding boxes which have y-coordinate < 365 or y-coordinate > 851.
        """
        clean_bboxes = []
        clean_classes_pred = []
        clean_scores = []
        for bb, cl, sc in zip(clean_bboxes_, clean_classes_pred_, clean_scores_):
            [x1, y1, x2, y2] = bb
            y_mean = (y2 + y1) / 2
            if y_mean < offset_y1_1 or y_mean > offset_y2_1:
                continue
            clean_bboxes.append(bb)
            clean_classes_pred.append(cl)
            clean_scores.append(sc)
        return clean_bboxes, clean_classes_pred, clean_scores

    @classmethod
    def reject_outliers(cls, data, m=1.5):
        """ This is not used. But we had a plan to reject some detections according to the scores. 
        Remove some detection which seems as an outlier in terms of score values.
        """
        return abs(data - np.mean(data)) < m * np.std(data)

    @classmethod
    def classification(cls, boxes, scores, labels, frame):
        boxes_ped, scores_ped, labels_ped = [], [], []
        boxes_car, scores_car, labels_car = [], [], []
        unsure_ped_boxes, unsure_ped_scores, unsure_ped_labels = [], [], []
        unsure_car_boxes, unsure_car_scores, unsure_car_labels = [], [], []

        keras_frame = frame[:, :, ::-1]

        for bb, sc, lb in zip(boxes, scores, labels):
            if lb == 1:
                if sc > cls.threshold_car:
                    boxes_car.append(bb)
                    scores_car.append(sc)
                    labels_car.append(lb)
                elif sc > 0.3:
                    unsure_car_boxes.append(bb)
                    unsure_car_scores.append(sc)
                    unsure_car_labels.append(lb)
            elif lb == 0:
                if sc > cls.threshold_pedestrian:
                    boxes_ped.append(bb)
                    scores_ped.append(sc)
                    labels_ped.append(lb)
                elif sc > 0.3:
                    unsure_ped_boxes.append(bb)
                    unsure_ped_scores.append(sc)
                    unsure_ped_labels.append(lb)

        if len(unsure_car_boxes) > 0:
            batch = np.zeros((4, 84, 84, 3))
            unsure_car_boxes = unsure_car_boxes[:4]
            unsure_car_scores = unsure_car_scores[:4]
            unsure_car_labels = unsure_car_labels[:4]
            unsure_car_batch = []

            pick_inds = []
            for ii, bb in enumerate(unsure_car_boxes):
                try:
                    img = keras_frame[int(bb[1]):int(bb[2]), int(bb[0]):int(bb[2])]
                    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
                    img = preprocess_input(img)
                    unsure_car_batch.append(img)
                    pick_inds.append(ii)
                except:
                    continue
            batch_size = len(unsure_car_batch)
            if batch_size > 0:
                unsure_car_batch = np.array(unsure_car_batch)
                batch[:batch_size, :, :, :] = unsure_car_batch
                unsure_car_preds = cls.car_classification_model.predict(batch)

                for jj in range(0, batch_size):
                    if unsure_car_preds[jj] > 0.8:
                        boxes_car.append(unsure_car_boxes[pick_inds[jj]])
                        scores_car.append(unsure_car_scores[pick_inds[jj]])
                        labels_car.append(unsure_car_labels[pick_inds[jj]])

                #
                # for jj, (bb, sc, lb, ii) in enumerate(zip(unsure_car_boxes[:batch_size], unsure_car_scores[:batch_size], unsure_car_labels[:batch_size], pick_inds)):
                #     if unsure_car_preds[jj] > 0.8:
                #         boxes_car.append(bb)
                #         scores_car.append(sc)
                #         labels_car.append(lb)

        if len(unsure_ped_boxes) > 0:
            batch = np.zeros((4, 128, 60, 3))
            unsure_ped_boxes = unsure_ped_boxes[:4]
            unsure_ped_scores = unsure_ped_scores[:4]
            unsure_ped_labels = unsure_ped_labels[:4]
            unsure_ped_batch = []

            pick_inds = []
            for ii, bb in enumerate(unsure_ped_boxes):
                try:
                    img = keras_frame[int(bb[1]):int(bb[2]), int(bb[0]):int(bb[2])]
                    img = cv2.resize(img, (60, 128), interpolation=cv2.INTER_CUBIC)
                    img = preprocess_input(img)
                    unsure_ped_batch.append(img)
                    pick_inds.append(ii)
                except:
                    continue
            batch_size = len(unsure_ped_batch)
            if batch_size > 0:
                unsure_ped_batch = np.array(unsure_ped_batch)
                batch[:batch_size, :, :, :] = unsure_ped_batch
                unsure_ped_preds = cls.pedestrian_classification_model.predict(batch)

                for jj in range(0, batch_size):
                    if unsure_ped_preds[jj] > 0.8:
                        boxes_ped.append(unsure_ped_boxes[pick_inds[jj]])
                        scores_ped.append(unsure_ped_scores[pick_inds[jj]])
                        labels_ped.append(unsure_ped_labels[pick_inds[jj]])

                # for jj, (bb, sc, lb, ii) in enumerate(zip(unsure_ped_boxes[:batch_size], unsure_ped_scores[:batch_size], unsure_ped_labels[:batch_size], pick_inds)):
                #     if unsure_ped_preds[jj] > 0.8:
                #         boxes_ped.append(bb)
                #         scores_ped.append(sc)
                #         labels_ped.append(lb)

        return boxes_car + boxes_ped, scores_car + scores_ped, labels_car + labels_ped

    @classmethod
    def model_inference(cls, frame, ii):
        """ Everything is processed here and this function is called from predict function.
        """
        # frame_darker = adjust_brightness(frame, -0.3)
        # frame_brighter = adjust_brightness(frame, 0.3)

        """ left crop """
        # img_inf2 = frame_brighter[cls.offset_y1_1:cls.offset_y2_1, :cls.offset_x2_1-cls.offset_x1_1]

        """ right crop """
        # img_inf3 = frame_brighter[cls.offset_y1_1:cls.offset_y2_1, cls.offset_x1_1 - cls.offset_x2_1:]
        # x_offset_3 = cls.w -img_inf3.shape[1]

        """ original image """
        img_inf0 = cls.preprocess_image(frame)
        scale0 = 1

        """ left crop """
        # img_inf2 = cls.preprocess_image(img_inf2)
        # img_inf2, scale2 = cls.resize_image(img_inf2, min_side=1216, max_side=1936)

        """ right crop """
        # img_inf3 = cls.preprocess_image(img_inf3)
        # img_inf3, scale3 = cls.resize_image(img_inf3, min_side=1216, max_side=1936)

        """ flip on x-axis """
        # img_inf4_ = cls.preprocess_image(frame_brighter)
        # img_inf4 = img_inf4_[:, ::-1, :]
        img_inf4 = img_inf0[:, ::-1, :]
        scale4 = 1

        # batch_size = 3:

        # img_inf5 = cls.preprocess_image(frame_brighter)
        # scale5 = 1

        # img_inf6 = cls.preprocess_image(frame_darker)
        # scale6 = 1

        # batch_list = [img_inf0, img_inf5, img_inf6]
        # batch_list = [img_inf0, img_inf2, img_inf3]
        batch_list = [img_inf0, img_inf4]
        # batch_list = [img_inf0, img_inf2, img_inf3, img_inf4, img_inf5, img_inf6]
        # batch_list = [img_inf0]
        boxes, scores, labels = cls.model.predict_on_batch(np.array(batch_list))

        # left_crop_order = 1  # 1
        # right_crop_order = 2  # 2
        flip_lr_order = 1  # 3
        # bright_order = 1  # 4
        # dark_order = 2  # 5

        boxes[0] = boxes[0] / scale0
        # boxes[left_crop_order] = boxes[left_crop_order] / scale2
        # boxes[right_crop_order] = boxes[right_crop_order] / scale3
        boxes[flip_lr_order] = boxes[flip_lr_order] / scale4
        # boxes[bright_order] = boxes[bright_order] / scale5
        # boxes[dark_order] = boxes[dark_order] / scale6

        boxes_0, scores_0, labels_0 = cls.classification(boxes[0], scores[0], labels[0], frame)
        clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = [], [], []
        clean_bboxes_car, clean_classes_pred_car, clean_scores_car = [], [], []

        for bbox_, score_, label_ in zip(boxes_0, scores_0, labels_0):
            [x1, y1, x2, y2] = bbox_
            width = x2 - x1
            height = y2 - y1
            if width * height < 1024:
                continue
            if label_ == 0:
                clean_bboxes_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                clean_classes_pred_pedestrian.append(label_)
                clean_scores_pedestrian.append(score_)
            elif label_ == 1:
                clean_bboxes_car.append([int(x1), int(y1), int(x2), int(y2)])
                clean_classes_pred_car.append(label_)
                clean_scores_car.append(score_)
            else:
                continue



        # for bbox_, score_, label_ in zip(boxes[0], scores[0], labels[0]):
        #     if label_ == -1:
        #         break
        #     if label_ == 0 and score_ < cls.threshold_pedestrian:
        #         continue
        #     if label_ == 1 and score_ < cls.threshold_car:
        #         continue
        #     [x1, y1, x2, y2] = bbox_
        #     width = x2 - x1
        #     height = y2 - y1
        #
        #     if width * height < 1024:
        #         continue
        #     if label_ == 0:
        #         clean_bboxes_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
        #         clean_classes_pred_pedestrian.append(label_)
        #         clean_scores_pedestrian.append(score_)
        #     elif label_ == 1:
        #         clean_bboxes_car.append([int(x1), int(y1), int(x2), int(y2)])
        #         clean_classes_pred_car.append(label_)
        #         clean_scores_car.append(score_)
        #     else:
        #         continue

        clean_bboxes_left_crop_pedestrian, clean_classes_pred_left_crop_pedestrian, clean_scores_left_crop_pedestrian = [], [], []
        clean_bboxes_left_crop_car, clean_classes_pred_left_crop_car, clean_scores_left_crop_car = [], [], []
        if cls.left_crop:  # left (center) crop
            for bbox_, score_, label_ in zip(boxes[left_crop_order], scores[left_crop_order], labels[left_crop_order]):
                if label_ == -1:
                    break
                if label_ == 0 and score_ < cls.threshold_pedestrian + cls.conf_score_bias:
                    continue
                if label_ == 1 and score_ < cls.threshold_car + cls.conf_score_bias:
                    continue

                [x1, y1, x2, y2] = bbox_
                y1 += cls.offset_y1_1
                y2 += cls.offset_y1_1
                width = x2 - x1
                height = y2 - y1
                if width * height < 1024:
                    continue

                if label_ == 0:
                    clean_bboxes_left_crop_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_left_crop_pedestrian.append(label_)
                    clean_scores_left_crop_pedestrian.append(score_)
                elif label_ == 1:
                    clean_bboxes_left_crop_car.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_left_crop_car.append(label_)
                    clean_scores_left_crop_car.append(score_)
                else:
                    continue

        clean_bboxes_right_crop_pedestrian, clean_classes_pred_right_crop_pedestrian, clean_scores_right_crop_pedestrian = [], [], []
        clean_bboxes_right_crop_car, clean_classes_pred_right_crop_car, clean_scores_right_crop_car = [], [], []
        if cls.right_crop:  # right (center) crop
            for bbox_, score_, label_ in zip(boxes[right_crop_order], scores[right_crop_order], labels[right_crop_order]):
                if label_ == -1:
                    break
                if label_ == 0 and score_ < cls.threshold_pedestrian + cls.conf_score_bias:
                    continue
                if label_ == 1 and score_ < cls.threshold_car + cls.conf_score_bias:
                    continue
                [x1, y1, x2, y2] = bbox_
                x1 += x_offset_3
                y1 += cls.offset_y1_1
                x2 += x_offset_3
                y2 += cls.offset_y1_1

                width = x2 - x1
                height = y2 - y1
                if width * height < 1024:
                    continue

                if label_ == 0:
                    clean_bboxes_right_crop_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_right_crop_pedestrian.append(label_)
                    clean_scores_right_crop_pedestrian.append(score_)
                elif label_ == 1:
                    clean_bboxes_right_crop_car.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_right_crop_car.append(label_)
                    clean_scores_right_crop_car.append(score_)
                else:
                    continue

        clean_bboxes_flip_lr_pedestrian, clean_classes_pred_flip_lr_pedestrian, clean_scores_flip_lr_pedestrian = [], [], []
        clean_bboxes_flip_lr_car, clean_classes_pred_flip_lr_car, clean_scores_flip_lr_car = [], [], []
        if cls.flip_lr:  # horizontal flip
            for bbox_, score_, label_ in zip(boxes[flip_lr_order], scores[flip_lr_order], labels[flip_lr_order]):
                if label_ == -1:
                    break
                if label_ == 0 and score_ < cls.threshold_pedestrian + cls.conf_score_bias:
                    continue
                if label_ == 1 and score_ < cls.threshold_car + cls.conf_score_bias:
                    continue
                [x1, y1, x2, y2] = bbox_
                x2_flip = cls.w - bbox_[0]
                x1_flip = cls.w - bbox_[2]

                x2 = x2_flip
                x1 = x1_flip

                width = x2 - x1
                height = y2 - y1
                if width * height < 1024:
                    continue

                if label_ == 0:
                    clean_bboxes_flip_lr_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_flip_lr_pedestrian.append(label_)
                    clean_scores_flip_lr_pedestrian.append(score_)
                elif label_ == 1:
                    clean_bboxes_flip_lr_car.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_flip_lr_car.append(label_)
                    clean_scores_flip_lr_car.append(score_)
                else:
                    continue

        clean_bboxes_bright_pedestrian, clean_classes_pred_bright_pedestrian, clean_scores_bright_pedestrian = [], [], []
        clean_bboxes_bright_car, clean_classes_pred_bright_car, clean_scores_bright_car = [], [], []
        if cls.bright_frame:
            for bbox_, score_, label_ in zip(boxes[bright_order], scores[bright_order], labels[bright_order]):
                if label_ == -1:
                    break
                if label_ == 0 and score_ < cls.threshold_pedestrian + cls.conf_score_bias:
                    continue
                if label_ == 1 and score_ < cls.threshold_car + cls.conf_score_bias:
                    continue
                [x1, y1, x2, y2] = bbox_

                width = x2 - x1
                height = y2 - y1
                if width * height < 1024:
                    continue

                if label_ == 0:
                    clean_bboxes_bright_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_bright_pedestrian.append(label_)
                    clean_scores_bright_pedestrian.append(score_)
                elif label_ == 1:
                    clean_bboxes_bright_car.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_bright_car.append(label_)
                    clean_scores_bright_car.append(score_)
                else:
                    continue

        clean_bboxes_dark_pedestrian, clean_classes_pred_dark_pedestrian, clean_scores_dark_pedestrian = [], [], []
        clean_bboxes_dark_car, clean_classes_pred_dark_car, clean_scores_dark_car = [], [], []
        if cls.dark_frame:
            for bbox_, score_, label_ in zip(boxes[dark_order], scores[dark_order], labels[dark_order]):
                if label_ == -1:
                    break
                if label_ == 0 and score_ < cls.threshold_pedestrian + cls.conf_score_bias:
                    continue
                if label_ == 1 and score_ < cls.threshold_car + cls.conf_score_bias:
                    continue

                [x1, y1, x2, y2] = bbox_

                width = x2 - x1
                height = y2 - y1
                if width * height < 1024:
                    continue

                if label_ == 0:
                    clean_bboxes_dark_pedestrian.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_dark_pedestrian.append(label_)
                    clean_scores_dark_pedestrian.append(score_)
                elif label_ == 1:
                    clean_bboxes_dark_car.append([int(x1), int(y1), int(x2), int(y2)])
                    clean_classes_pred_dark_car.append(label_)
                    clean_scores_dark_car.append(score_)
                else:
                    continue

        """ merge: overall + flip_lr """
        if len(clean_bboxes_flip_lr_pedestrian) > 0:
            clean_bboxes_pedestrian += clean_bboxes_flip_lr_pedestrian
            clean_classes_pred_pedestrian += clean_classes_pred_flip_lr_pedestrian
            clean_scores_pedestrian += clean_scores_flip_lr_pedestrian
            clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = cls.apply_local_nms(clean_bboxes_pedestrian,
                                                                                                                  clean_classes_pred_pedestrian,
                                                                                                                  clean_scores_pedestrian)
        if len(clean_bboxes_flip_lr_car) > 0:
            clean_bboxes_car += clean_bboxes_flip_lr_car
            clean_classes_pred_car += clean_classes_pred_flip_lr_car
            clean_scores_car += clean_scores_flip_lr_car
            clean_bboxes_car, clean_classes_pred_car, clean_scores_car = cls.apply_local_nms(clean_bboxes_car,
                                                                                             clean_classes_pred_car,
                                                                                             clean_scores_car)

        """ merge: overall + left_crop """
        if len(clean_bboxes_left_crop_pedestrian) > 0:
            clean_bboxes_pedestrian += clean_bboxes_right_crop_pedestrian
            clean_classes_pred_pedestrian += clean_classes_pred_right_crop_pedestrian
            clean_scores_pedestrian += clean_scores_right_crop_pedestrian
            clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = cls.apply_local_nms(clean_bboxes_pedestrian,
                                                                                                                  clean_classes_pred_pedestrian,
                                                                                                                  clean_scores_pedestrian)
        if len(clean_bboxes_left_crop_pedestrian) > 0:
            clean_bboxes_car += clean_bboxes_right_crop_car
            clean_classes_pred_car += clean_classes_pred_right_crop_car
            clean_scores_car += clean_scores_right_crop_car
            clean_bboxes_car, clean_classes_pred_car, clean_scores_car = cls.apply_local_nms(clean_bboxes_car,
                                                                                             clean_classes_pred_car,
                                                                                             clean_scores_car)

        """ merge: overall + right_crop """
        if len(clean_bboxes_right_crop_pedestrian) > 0:
            clean_bboxes_pedestrian += clean_bboxes_left_crop_pedestrian
            clean_classes_pred_pedestrian += clean_classes_pred_left_crop_pedestrian
            clean_scores_pedestrian += clean_scores_left_crop_pedestrian
            clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = cls.apply_local_nms(clean_bboxes_pedestrian,
                                                                                                                  clean_classes_pred_pedestrian,
                                                                                                                  clean_scores_pedestrian)
        if len(clean_bboxes_right_crop_car) > 0:
            clean_bboxes_car += clean_bboxes_left_crop_car
            clean_classes_pred_car += clean_classes_pred_left_crop_car
            clean_scores_car += clean_scores_left_crop_car
            clean_bboxes_car, clean_classes_pred_car, clean_scores_car = cls.apply_local_nms(clean_bboxes_car,
                                                                                             clean_classes_pred_car,
                                                                                             clean_scores_car)

        """ merge: overall + bright """
        if len(clean_bboxes_bright_pedestrian) > 0:
            clean_bboxes_pedestrian += clean_bboxes_bright_pedestrian
            clean_classes_pred_pedestrian += clean_classes_pred_bright_pedestrian
            clean_scores_pedestrian += clean_scores_bright_pedestrian
            clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = cls.apply_local_nms(clean_bboxes_pedestrian,
                                                                                                                  clean_classes_pred_pedestrian,
                                                                                                                  clean_scores_pedestrian)
        if len(clean_bboxes_bright_car) > 0:
            clean_bboxes_car += clean_bboxes_bright_car
            clean_classes_pred_car += clean_classes_pred_bright_car
            clean_scores_car += clean_scores_bright_car

            clean_bboxes_car, clean_classes_pred_car, clean_scores_car = cls.apply_local_nms(clean_bboxes_car,
                                                                                             clean_classes_pred_car,
                                                                                             clean_scores_car)

        """ merge: overall + dark """
        if len(clean_bboxes_dark_pedestrian) > 0:
            clean_bboxes_pedestrian += clean_bboxes_dark_pedestrian
            clean_classes_pred_pedestrian += clean_classes_pred_dark_pedestrian
            clean_scores_pedestrian += clean_scores_dark_pedestrian
            clean_bboxes_pedestrian, clean_classes_pred_pedestrian, clean_scores_pedestrian = cls.apply_local_nms(clean_bboxes_pedestrian,
                                                                                                                  clean_classes_pred_pedestrian,
                                                                                                                  clean_scores_pedestrian)
        if len(clean_bboxes_dark_car) > 0:
            clean_bboxes_car += clean_bboxes_dark_car
            clean_classes_pred_car += clean_classes_pred_dark_car
            clean_scores_car += clean_scores_dark_car
            clean_bboxes_car, clean_classes_pred_car, clean_scores_car = cls.apply_local_nms(clean_bboxes_car,
                                                                                             clean_classes_pred_car,
                                                                                             clean_scores_car)

        """ global non max suppression """
        if cls.left_crop or cls.right_crop or cls.flip_lr or cls.dark_frame or cls.bright_frame:
            pick_inds_pedestrian = cls.non_max_suppression_with_scores(clean_bboxes_pedestrian, probs=clean_scores_pedestrian,
                                                                       overlapThresh=cls.pedestrian_nms_thr)

            clean_bboxes_pedestrian_nms = list(clean_bboxes_pedestrian[i] for i in pick_inds_pedestrian)
            clean_classes_pred_pedestrian_nms = list(clean_classes_pred_pedestrian[i] for i in pick_inds_pedestrian)
            clean_scores_pedestrian_nms = list(clean_scores_pedestrian[i] for i in pick_inds_pedestrian)

            if cls.apply_adaptive_pedestrian_nms:
                if len(clean_scores_pedestrian_nms) > 8:
                    pick_inds_pedestrian = cls.non_max_suppression_with_scores(clean_bboxes_pedestrian,
                                                                               probs=clean_scores_pedestrian,
                                                                               overlapThresh=cls.pedestrian_nms_thr * 0.8)
                    clean_bboxes_pedestrian_nms = list(clean_bboxes_pedestrian[i] for i in pick_inds_pedestrian)
                    clean_classes_pred_pedestrian_nms = list(clean_classes_pred_pedestrian[i] for i in pick_inds_pedestrian)
                    clean_scores_pedestrian_nms = list(clean_scores_pedestrian[i] for i in pick_inds_pedestrian)

            pick_inds_car = cls.non_max_suppression_with_scores(clean_bboxes_car, probs=clean_scores_car, overlapThresh=cls.car_nms_thr)
            clean_bboxes_car_nms = list(clean_bboxes_car[i] for i in pick_inds_car)
            clean_classes_pred_car_nms = list(clean_classes_pred_car[i] for i in pick_inds_car)
            clean_scores_car_nms = list(clean_scores_car[i] for i in pick_inds_car)

            clean_bboxes = clean_bboxes_pedestrian_nms + clean_bboxes_car_nms
            clean_classes_pred = clean_classes_pred_pedestrian_nms + clean_classes_pred_car_nms
            clean_scores = clean_scores_pedestrian_nms + clean_scores_car_nms
        else:
            clean_bboxes = clean_bboxes_pedestrian + clean_bboxes_car
            clean_classes_pred = clean_classes_pred_pedestrian + clean_classes_pred_car
            clean_scores = clean_scores_pedestrian + clean_scores_car

        if cls.apply_heuristic_post_processing:
            clean_bboxes, clean_classes_pred, clean_scores = cls.apply_heuristics(clean_bboxes,
                                                                                  clean_classes_pred,
                                                                                  clean_scores,
                                                                                  cls.offset_y1_1,
                                                                                  cls.offset_y2_1)

        pedestrian_list = []
        car_list = []
        for bbox, score, label in zip(clean_bboxes, clean_scores, clean_classes_pred):
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            if area < 1024:
                continue
            if label == 0:  # Pedestrian
                pedestrian_list.append({"box2d": bbox, "score": score})
            elif label == 1:  # Car
                # if width / float(height) < 0.9 and score < 0.9:
                #     continue
                car_list.append({"box2d": bbox, "score": score})
            else:
                print("Irrelevant class detected: {}".format(label))
                continue
        current_frame = {"Car": car_list, "Pedestrian": pedestrian_list}
        pred_tracking = cls.tracker.assign_ids(current_frame, frame)

        return pred_tracking

    @classmethod
    def predict(cls, input):
        """Predict method
 
        Args:
            input (str): path to the video file you want to make inference from
 
        Returns:
            dict: Inference for the given input.
                format:
                    - filename []:
                        - category_1 []:
                            - id: int
                            - box2d: [left, top, right, bottom]
                        ...
        Notes:
            - The categories for testing are "Car" and "Pedestrian".
              Do not include other categories in the prediction you will make.
            - If you do not want to make any prediction in some frames,
              just write "prediction = {}" in the prediction of the frame in the sequence(in line 65 or line 67).
        """
        print("Predicting file {}".format(input))

        predictions = []
        cap = cv2.VideoCapture(input)
        cls.w, cls.h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        wc = cls.w // 2
        hc = cls.h // 2
        cls.offset_x1_1 = int(wc - int(cls.w * cls.scales[0]))
        cls.offset_y1_1 = int(hc - int(cls.h * cls.scales[0]))
        cls.offset_x2_1 = int(wc + int(cls.w * cls.scales[0]))
        cls.offset_y2_1 = int(hc + int(cls.h * cls.scales[0]))

        cls.tracker = Tracker((cls.w, cls.h))
        prev_prediction = {}
        fname = os.path.basename(input)
        ii = 0
        break_time = time.time()
        while True:
            start_time = time.time()
            ii += 1
            ret, frame = cap.read()
            if not ret:
                break
            try:
                if cls.model is not None:
                    prediction = cls.model_inference(frame, ii)
                    if prediction is None:
                        prediction = prev_prediction
                    predictions.append(prediction)
                    prev_prediction = copy.copy(prediction)
                else:
                    prediction = {}
                    predictions.append(prediction)
            except:
                predictions.append(prev_prediction)

            if (ii % 10 == 0) :
                print("Frames processed : {} ({})".format(ii, start_time - break_time))
                break_time = time.time()
        cap.release()
        if cls.reassign_id_pedestrian:
            predictions = cls.filter_predictions_switch(predictions)
        else:
            predictions = cls.filter_predictions(predictions)
        end_time = time.time()
        # print("[PERFORMANCE] Video {} Frame {} Total_Time     = {}".format(fname, ii, end_time - start_time))
        # print("-----------------------------")
        return {fname: predictions}
    
    @classmethod
    def filter_predictions(cls, data):
        """ Here we apply the filter of objects whose ID doesn’t appear more 3 times on the frame. 
        This function is used to follow competition rules. """
        id_count = defaultdict(lambda: defaultdict(int))
        for sequence in data:
            for c in sequence:
                ids = [ item['id'] for item in sequence[c] ]
                for id in ids:
                    id_count[c][id] = id_count[c][id] + 1

        for sequence in data:
            for c in sequence:
                sequence[c] = list(filter(lambda i: id_count[c][i['id']] > cls.min_no_of_frames, sequence[c]))

        return data

    @classmethod
    def filter_predictions_switch(cls, data, dummy_id = 7777):
        """Switch ID of object_id if appears less than 3 time in the seq."""
        # After experiment only improve score for pedestrian so only on pedestrian
        id_count = defaultdict(lambda: defaultdict(int))
        re_assign_objects = 0
        start = time.time()

        # Enum all ID into dict
        for sequence in data:
            for c in sequence:
                ids = [ item['id'] for item in sequence[c] ]
                for id in ids:
                    id_count[c][id] = id_count[c][id] + 1

        for sequence in data:
            for c in sequence:
                if c == "Pedestrian":
                    # Filter out object to be reassign and find id
                    id_to_change = []
                    obj_to_reassign = list(filter(lambda i: id_count[c][i['id']] < cls.min_no_of_frames, sequence[c]))
                    if len(obj_to_reassign) > 0:
                        re_assign_objects += len(obj_to_reassign)
                        for _obj in obj_to_reassign:
                            id_to_change.append(_obj['id'])
                    
                    # Switch all the id from the list:
                    if id_to_change:
                        dummy_index = 0
                        for obj in sequence[c]:
                            if obj['id'] in id_to_change:
                                obj['id'] = cls.dummy_id_range[dummy_index]
                                dummy_index += 1
                else:
                    # Decrease score on Car so simply removing
                    sequence[c] = list(filter(lambda i: id_count[c][i['id']] > cls.min_no_of_frames, sequence[c]))

        print("Post processing:")
        print("  Time: {}".format(time.time() - start))
        print("  Number of Reasign: {}".format(re_assign_objects))

        return data
