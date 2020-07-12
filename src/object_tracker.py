# -*- coding: utf-8 -*-
# -----------------------------------------------------------
#  Object tracking class to track objects and assign IDs.
#  Released under Apache License 2.0
#  Email: machine.learning.tokyo@gmail.com
# -----------------------------------------------------------
import cv2
import os
import numpy as np
import copy
import json
import time
import shutil
from queue import Queue
from glob import glob
from argparse import ArgumentParser
from statistics import mean
from concurrent.futures import ProcessPoolExecutor


class Tracker:
    def __init__(self, image_size):
        self.init_predictions = {'Car': [], 'Pedestrian': []}
        self.predictions = [self.init_predictions]; # past predictions in the format {'id': id, 'box2d': [x1, y1, x2, y2], 'mv': [vx, vy], 'scale': [sx, sy], 'occlusion': number_of_occlusions, 'image': image}
        self.image_size = image_size # input frame resolution: (width, height)
        self.max_occ_frames = 12 # max number of frames for which the tracker keeps occluded objects
        self.frame_out_thresh = {'Car': 0.2, 'Pedestrian': 0.2}
        self.box_area_thresh = 1024 # ignore bounding boxes with area less than this threshold(px)
        self.last_id = -1 # the biggest ID already assigned so far
        self.total_cost = 0
        self.max_frame_in = {'Car': 6, 'Pedestrian': 7}

        # cost weights for hungarian matching
        self.cost_weight = {'Car': [0.135, 1.44], 'Pedestrian': [0.0375, 1.14]} # [a, b]: a is for box distance, b is for box size difference
        self.sim_weight = {'Car': 1.13, 'Pedestrian': 0.99} # cost for two boxes' image similarity
        self.occ_weight = {'Car': 0.85, 'Pedestrian': 0.7} # cost to detect a object in the previous frame as occluded
        self.frame_in_weight = {'Car': 0.14, 'Pedestrian': 0.43} # cost to detect a object as in the current frame as newly framed in

    def iou(self, a, b):
        if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
            return 0.0
        x, y = max(a[0], b[0]), max(a[1], b[1])
        w, h = min(a[2], b[2]) - x, min(a[3], b[3]) - y
        area_i = 0 if w<0 or h<0 else w*h
        area_a, area_b = (a[2]-a[0]) * (a[3]-a[1]), (b[2]-b[0]) * (b[3]-b[1])
        area_u = area_a + area_b - area_i
        return area_i / (area_u + 1e-6)


    def get_bb_image(self, image, bb):
        bb = [max(0, bb[0]), max(0, bb[1]), min(self.image_size[0]-1, bb[2]), min(self.image_size[1]-1, bb[3])]
        bb = [min(bb[0], bb[2]), min(bb[1], bb[3]), max(bb[0], bb[2]), max(bb[1], bb[3])]
        im = image[int(bb[1]):int(bb[3]+1), int(bb[0]):int(bb[2]+1), :]
        return im

    def smooth_image(self, image):
        im = cv2.bilateralFilter(image.copy(), 7, 40, 40)
        im = cv2.bilateralFilter(im, 7, 40, 40)
        return im

    def get_hist(self, image, hist_mask):
        image = cv2.cvtColor(cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2HSV)
        im1 = image[:64, :64, :]
        im2 = image[:64, 64:, :]
        im3 = image[64:, :64, :]
        im4 = image[64:, 64:, :]
        im5 = image[32:-32, 32:-32]
        hist1 = [cv2.calcHist([im1], [c], hist_mask[:64, :64], [64], [0, 256]) for c in range(3)]
        hist2 = [cv2.calcHist([im2], [c], hist_mask[:64, 64:], [64], [0, 256]) for c in range(3)]
        hist3 = [cv2.calcHist([im3], [c], hist_mask[64:, :64], [64], [0, 256]) for c in range(3)]
        hist4 = [cv2.calcHist([im4], [c], hist_mask[64:, 64:], [64], [0, 256]) for c in range(3)]
        hist5 = [cv2.calcHist([im5], [c], hist_mask[32:-32, 32:-32], [64], [0, 256]) for c in range(3)]
        hist = hist1 + hist2 + hist3 + hist4 + hist5
        return hist


    def get_similality(self, hist1, hist2):
        hist_score = [cv2.compareHist(hist1[c], hist2[c], cv2.HISTCMP_CORREL) for c in range(len(hist1))]
        hist_score = mean(hist_score)
        return hist_score


    def calculate_cost(self, box1, box2, hist1, hist2, score1, score2, cls='Car'):
        """ Calculate cost used for Hungarian matching: histogram, object distance and size differences. """
        w1, h1 = box1[2]-box1[0]+1, box1[3]-box1[1]+1
        w2, h2 = box2[2]-box2[0]+1, box2[3]-box2[1]+1

        # compare the RGB histograms of two given bbox images
        hist_score = self.get_similality(hist1, hist2)
        if score1>=0:
            r = min(score1/(score2+1e-12), score2/(score1+1e-12))
            hist_score *= pow(r, 0.3)

        cnt1 = [box1[0]+w1/2, box1[1]+h1/2]
        cnt2 = [box2[0]+w2/2, box2[1]+h2/2]
        alpha = ((cnt1[0]-cnt2[0])/(w1+w2))**2 + ((cnt1[1]-cnt2[1])/(h1+h2))**2 # cost for distance between two objects
        beta = (w1+w2)/(2*np.sqrt(w1*w2)) * (h1+h2)/(2*np.sqrt(h1*h2)) # cost for size difference between two objects

        cost = pow(alpha, self.cost_weight[cls][0]) * pow(beta, self.cost_weight[cls][1]) * pow(2, (0.5-hist_score)*self.sim_weight[cls])
        return cost


    # find the optimal matching between objects in the previous frame (+occluded objects in the past frames) and objects in the current frame
    # using hungarian algorithm for maximum weighted matching
    def hungarian_match(self, preds1, preds2, cls='Car'):
        """ Find the optimal matching between objects in the previous frame (and occluded objects in the past frames) and 
        objects in the current frame by using hungarian algorithm for maximum weighted matching.
        """
        n1 = len(preds1) # number of objects in the previous frame
        n2 = len(preds2) # number of objects in the current frame

        # calculate the costs for each combination of objects between the previous frame and the current frame in advance
        match_costs = [[0]*n2 for _ in range(n1)]
        hist1s = [preds1[i]['hist'] for i in range(n1)]
        hist2s = [preds2[i]['hist'] for i in range(n2)]

        for i in range(n1):
            for j in range(n2):
                match_costs[i][j] = self.calculate_cost(preds1[i]['box2d'], preds2[j]['box2d'], hist1s[i], hist2s[j], preds1[i]['score'], preds2[j]['score'], cls)
        best_box_map = []
        min_cost = 1e16
        no_update = 0
        for n_occ in range(max(n1-n2, 0), max(n1-n2+self.max_frame_in[cls]+1, min(n2, self.max_frame_in[cls]))):
            prev_min_cost = min_cost
            n_match = n1 - n_occ
            if n_match>n2 or n_match<0:
                continue
            n_frame_in = n2-n_match
            fcosts = [c[:] for c in match_costs]
            for i in range(n_frame_in):
                fcosts.append([self.frame_in_weight[cls]]*n2)
            for i in range(n_occ):
                for j in range(len(fcosts)):
                    if j<n1:
                        p1 = preds1[j]
                        # decrease occlusion cost for a previously occluded object if it's covered by non-occluded objects
                        if p1['occlusion']>0:
                            bb1 = p1['box2d']
                            flags = np.zeros((int(bb1[3]-bb1[1]+1), int(bb1[2]-bb1[0]+1)), np.bool)
                            for k in range(n1):
                                if k!=j:
                                    p2 = preds1[k]
                                    if p2['occlusion']==0:
                                        bb2 = p2['box2d']
                                        x1 = max(bb1[0], bb2[0]) - bb1[0]
                                        y1 = max(bb1[1], bb2[1]) - bb1[1]
                                        x2 = min(bb1[2], bb2[2]) - bb1[0]
                                        y2 = min(bb1[3], bb2[3]) - bb1[1]
                                        flags[int(y1):int(y2), int(x1):int(x2)] = True
                            occ_rate = np.count_nonzero(flags) / (flags.shape[0]*flags.shape[1])
                            fcosts[j].append(self.occ_weight[cls]*(1-occ_rate))
                        else:
                            fcosts[j].append(self.occ_weight[cls])
                    else:
                        fcosts[j].append(self.occ_weight[cls])
            tcosts = np.array(fcosts)
            tcosts = (tcosts*100000).astype(np.int)

            # hungarian algorithm
            if len(tcosts)>0:
                # step1
                tcosts -= tcosts.min(axis=1)[:, None]
                tcosts -= tcosts.min(axis=0)
                marks = np.zeros_like(tcosts)
                prev_marks = copy.deepcopy(marks)
                while not (((marks==1).sum(axis=0)==1).all() and ((marks==1).sum(axis=1)==1).all()):
                    marks = np.zeros_like(tcosts)
                    prev_tcosts = tcosts.copy()

                    # step2
                    while True:
                        while True:
                            updated1 = False
                            zero_costs = tcosts==0
                            for i in range(tcosts.shape[0]):
                                if (marks[i]!=1).all() and np.count_nonzero(np.logical_and(zero_costs[i], marks[i]==0))==1:
                                    idx = np.where(np.logical_and(zero_costs[i], marks[i]==0))[0][0]
                                    marks[:, idx][tcosts[:, idx]==0] = -1
                                    marks[i, :][tcosts[i, :]==0] = -1
                                    marks[i, idx] = 1
                                    updated1 = True
                            for i in range(tcosts.shape[1]):
                                if (marks[:, i]!=1).all() and np.count_nonzero(np.logical_and(zero_costs[:, i], marks[:, i]==0))==1:
                                    idx = np.where(np.logical_and(zero_costs[:, i], marks[:, i]==0))[0][0]
                                    marks[idx, :][tcosts[idx, :]==0] = -1
                                    marks[:, i][tcosts[:, i]==0] = -1
                                    marks[idx, i] = 1
                                    updated1 = True
                            if not updated1:
                                break
                        updated2 = False
                        unmarked_zeros = np.logical_and(zero_costs, marks==0)
                        nr_unmarked_zeros = np.count_nonzero(unmarked_zeros, axis=1)
                        indices = np.where(nr_unmarked_zeros>0)[0]
                        rows = indices[nr_unmarked_zeros[indices]==nr_unmarked_zeros[indices].min()] if indices.shape[0]>0 else np.array([])
                        nc_unmarked_zeros = np.count_nonzero(unmarked_zeros, axis=0)
                        indices = np.where(nc_unmarked_zeros>0)[0]
                        cols = indices[nc_unmarked_zeros[indices]==nc_unmarked_zeros[indices].min()] if indices.shape[0]>0 else np.array([])
                        if rows.shape[0]>0 or cols.shape[0]>0:
                            cands = []
                            for r in rows:
                                rcs = np.where(np.logical_and(zero_costs[r, :], marks[r, :]==0))[0]
                                for rc in rcs:
                                    cands.append((r, rc, nc_unmarked_zeros[rc]))
                            for c in cols:
                                crs = np.where(np.logical_and(zero_costs[:, c], marks[:, c]==0))[0]
                                for cr in crs:
                                    cands.append((cr, c, nr_unmarked_zeros[cr]))
                            cands.sort(key=lambda cand: cand[2])
                            r, c = cands[0][0], cands[0][1]
                            marks[r, :][tcosts[r, :]==0] = -1
                            marks[:, c][tcosts[:, c]==0] = -1
                            marks[r, c] = 1
                            updated2 = True
                        if not updated2:
                            break

                    # step3
                    row_flags = np.zeros(tcosts.shape[0], np.bool)
                    col_flags = np.zeros(tcosts.shape[1], np.bool)
                    row_queue = Queue()
                    col_queue = Queue()
                    for i in range(tcosts.shape[0]):
                        if np.count_nonzero(marks[i]==1)==0:
                            row_queue.put(i)
                            row_flags[i] = True
                    while not (row_queue.empty() and col_queue.empty()):
                        while not row_queue.empty():
                            row = row_queue.get()
                            cols = np.where(np.logical_and(marks[row, :]==-1, np.logical_not(col_flags)))[0]
                            for col in cols:
                                col_queue.put(col)
                                col_flags[col] = True
                        while not col_queue.empty():
                            col = col_queue.get()
                            rows = np.where(np.logical_and(marks[:, col]==1, np.logical_not(row_flags)))[0]
                            for row in rows:
                                row_queue.put(row)
                                row_flags[row] = True

                    # step4
                    if len(tcosts[row_flags==1])>0:
                        tmp = tcosts[row_flags==1]
                        if len(tcosts[row_flags==1][np.tile(col_flags==0, (len(tmp), 1))])>0:
                            mask_min = tcosts[row_flags==1][np.tile(col_flags==0, (len(tmp), 1))].min()
                        else:
                            mask_min = 0
                    else:
                        mask_min = 0
                    if len(tcosts[row_flags==1])>0:
                        mask = np.logical_and(np.tile(row_flags[:, np.newaxis], [1, col_flags.shape[0]]), np.tile(np.logical_not(col_flags), [row_flags.shape[0], 1]))
                        tcosts[mask] -= mask_min
                    if len(tcosts[row_flags==0])>0:
                        mask = np.logical_and(np.tile(np.logical_not(row_flags)[:, np.newaxis], [1, col_flags.shape[0]]), np.tile(col_flags, [row_flags.shape[0], 1]))
                        tcosts[mask] += mask_min
                    if (prev_tcosts==tcosts).all() and (prev_marks==marks).all():
                        break
                    prev_marks = marks.copy()

            # create ID mapping to return, using hungarian matching result
            box_map = []
            cost = 0
            indices = set(range(n2+n_occ))
            term = False
            for i in range(tcosts.shape[0]):
                tmp = np.where(marks[i]==1)[0]
                if len(tmp)==1:
                    idx = tmp[0]
                    indices.remove(idx)
                else:
                    term = True
            for i in range(tcosts.shape[0]):
                tmp = np.where(marks[i]==1)[0]
                if len(tmp)==1:
                    idx = np.where(marks[i]==1)[0][0]
                else:
                    idx = list(indices)[0]
                    indices.remove(idx)
                if i<n1:
                    box_map.append(idx if idx<n2 else -1)
                cost += fcosts[i][idx]
            if cost < min_cost:
                min_cost = cost
                best_box_map = box_map

            if min_cost==prev_min_cost:
                no_update += 1
            else:
                no_update = 0

            if no_update>=4:
                break

        return best_box_map, min_cost


    def assign_ids(self, pred, image): # {'Car': [{'box2d': [x1, y1, x2, y2], 'score': s}], 'Pedestrian': [{'box2d': [x1, y1, x2, y2], 'score': s}]}
        """ Main function of our tracker that manages to retrieve information from previous frames, 
        extracts features from the different objects from the new frame (position, size, HSV histogram, …), 
        predicts next position, match current frame object with history, and finally assign ids.
        """
        hist_size = 128
        hist_mask = {'Car': np.ones((hist_size, hist_size), np.uint8), 'Pedestrian': np.ones((hist_size, hist_size), np.uint8)}
        for y in range(hist_size//2):
            hist_mask['Pedestrian'][y, :hist_size//8*3-y*3//4] = 0
            hist_mask['Pedestrian'][y, hist_size//8*5+y*3//4:] = 0
            hist_mask['Pedestrian'][hist_size//2+y, :y*3//4] = 0
            hist_mask['Pedestrian'][hist_size//2+y, hist_size-y*3//4:] = 0
        for y in range(hist_size//4):
            hist_mask['Car'][y, :hist_size//4-y] = 0
            hist_mask['Car'][y, hist_size//4*3+y:] = 0
            hist_mask['Car'][hist_size//4*3+y, :y] = 0
            hist_mask['Car'][hist_size//4*3+y, hist_size-y:] = 0

        pred = copy.deepcopy(pred)
        for cls, boxes in pred.items():

            # get last predictions
            if cls not in pred:
                pred[cls] = self.init_predictions[cls]
            if cls not in self.predictions[-1]:
                last_preds = self.init_predictions[cls]
            else:
                last_preds = self.predictions[-1][cls]

            adjusted_preds = [] # bboxes predicted from bboxes in the last frame, using velocity of position/size
            n_frame_out = 0


            for p in last_preds:
                bb = p['box2d']
                cnt = ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)

                # estimate speed (motion vector) of each object and predict next position
                # using up to 16 past frames
                cnts = [cnt]
                n_empty = 0
                for i in range(2, 13):
                    if len(self.predictions)>=i:
                        try:
                            past_pred = self.predictions[-i][cls]
                        except:
                            if n_empty>=1 or i>=4:
                                break
                            n_empty += 1
                            cnts.append(None)
                            continue
                        if p['id'] in map(lambda pp: pp['id'], past_pred):
                            bb = list(filter(lambda pp: pp['id']==p['id'], past_pred))[0]['box2d']
                            cnts.append(((bb[0]+bb[2])/2, (bb[1]+bb[3])/2))
                        else:
                            if n_empty>=1 or i>=4:
                                break
                            n_empty += 1
                            cnts.append(None)
                            continue
                    else:
                        break

                n_sample = len(list(filter(lambda c: c is not None, cnts)))

                # if an object is not in previous frames and it's on the edge of a frame,
                # estimate the previous position
                if n_sample==1:
                    w = bb[2] - bb[0]
                    h = bb[3] - bb[1]
                    mx = min(cnt[0], self.image_size[0]-cnt[0])
                    my = min(cnt[1], self.image_size[1]-cnt[1])
                    if mx<w*1.0 and my<h*1.0:
                        x = 0 if cnt[0]<self.image_size[0]-cnt[0] else self.image_size[0]-1
                        y = 0 if cnt[1]<self.image_size[1]-cnt[1] else self.image_size[1]-1
                        cnts.append((x, y))
                    elif mx<w*1.0:
                        x = 0 if cnt[0]<self.image_size[0]-cnt[0] else self.image_size[0]-1
                        y = cnt[1]
                        cnts.append((x, y))
                    elif my<h*1.0:
                        x = cnt[0]
                        y = 0 if cnt[1]<self.image_size[1]-cnt[1] else self.image_size[1]-1
                        cnts.append((x, y))

                if n_sample>=2:
                    cnts = cnts[::-1]
                    while cnts[-1] is None:
                        cnts = cnts[:-1]
                    ts = []
                    for i in range(len(cnts)):
                        if cnts[i] is not None:
                            ts.append(i)
                    if n_sample<=3:
                        # linear regression
                        xs = [cnt[0] for cnt in cnts if cnt is not None]
                        ys = [cnt[1] for cnt in cnts if cnt is not None]
                        n = len(cnts)
                        xcs = np.polyfit(ts, xs, 1)
                        ycs = np.polyfit(ts, ys, 1)
                        x = xcs[0] * n + xcs[1]
                        y = ycs[0] * n + ycs[1]
                        cnt = [x, y]
                    else:
                        # quadratic regression
                        xs = [cnt[0] for cnt in cnts if cnt is not None]
                        ys = [cnt[1] for cnt in cnts if cnt is not None]
                        n = len(cnts)
                        xcs = np.polyfit(ts, xs, 2)
                        ycs = np.polyfit(ts, ys, 2)
                        x = xcs[0] * n**2 + xcs[1] * n + xcs[2]
                        y = ycs[0] * n**2 + ycs[1] * n + ycs[2]
                        cnt = [x, y]

                # estimate scaling speed of each object and predict next size
                scale = p['scale']
                bb = p['box2d']
                w = bb[2]-bb[0]+1
                h = bb[3]-bb[1]+1
                sw = w * scale[0]
                sh = h * scale[1]
                x1 = int(cnt[0] - sw/2)
                x2 = int(cnt[0] + sw/2)
                y1 = int(cnt[1] - sh/2)
                y2 = int(cnt[1] + sh/2)
                box2d = [max(0, x1), max(0, y1), min(self.image_size[0]-1, x2), min(self.image_size[1]-1, y2)]
                box2d = [min(box2d[0], box2d[2]), min(box2d[1], box2d[3]), max(box2d[0], box2d[2]), max(box2d[1], box2d[3])]

                # filter out objects with area less than the threshold
                area = (box2d[2]-box2d[0]+1) * (box2d[3]-box2d[1]+1)
                if area<self.box_area_thresh:
                    continue

                # filter out objects that are predicted to have gone outside of the frame
                box2d_inside = [max(0, box2d[0]), max(0, box2d[1]), min(self.image_size[0]-1, box2d[2]), min(self.image_size[1]-1, box2d[3])]
                area_inside = (box2d_inside[2]-box2d_inside[0]+1) * (box2d_inside[3]-box2d_inside[1]+1)
                if area_inside <=area*self.frame_out_thresh[cls]:
                    n_frame_out += 1
                    continue
                adjusted_pred = {'id': p['id'], 'box2d': box2d_inside, 'score': p['score'], 'mv': p['mv'], 'scale': p['scale'], 'occlusion': p['occlusion'], 'image': p['image'], 'hist': p['hist']}
                adjusted_preds.append(adjusted_pred)

                # check surrounding area to find interpolation candidates
                iw = box2d_inside[2] - box2d_inside[0]
                ih = box2d_inside[3] - box2d_inside[1]
                r = min(iw/(ih+1e-10), ih/(iw+1e-10))
                size_flag = r>0.2
                if cls=='Car':
                    size_flag = size_flag and iw/(ih+1e-10)>0.5
                else:
                    size_flag = size_flag and iw/(ih+1e-10)<2.0

                if p['occlusion']==0 and area_inside>1024*1.2 and size_flag:
                    ew = w*0.6 if cls=='Car' else w*0.5
                    eh = h*0.6 if cls=='Car' else h*0.5
                    ex_bb = (cnt[0]-ew, cnt[1]-eh, cnt[0]+ew, cnt[1]+eh)
                    bb_image = self.get_bb_image(image, box2d_inside)
                    missing = True
                    for box in boxes:
                        tmp_bb = copy.deepcopy(box['box2d'])
                        curr_iou = self.iou(ex_bb, tmp_bb)
                        iou_thresh = 0.1 if cls=='Pedestrian' else 0
                        if curr_iou>iou_thresh:
                            missing = False
                            break
                    if missing:
                        hist = self.get_hist(bb_image, hist_mask[cls])
                        similarity = self.get_similality(hist, p['hist'])
                        checked = set([(0, 0)])
                        pos = (0, 0)
                        sx = max(1, w//8)
                        sy = max(1, h//8)
                        prev_similarity = similarity
                        count = 0
                        while True:
                            updated = False
                            for tpos in [(pos[0]-sx, pos[1]), (pos[0], pos[1]-sy), (pos[0]+sx, pos[1]), (pos[0], pos[1]+sy)]:
                                if tpos in checked:
                                    continue
                                tbox2d_inside = [box2d_inside[0]+tpos[0], box2d_inside[1]+tpos[1], box2d_inside[2]+tpos[0], box2d_inside[3]+tpos[1]]
                                if tbox2d_inside[0]<0 or tbox2d_inside[2]>=self.image_size[0] or tbox2d_inside[1]<0 or tbox2d_inside[3]>=self.image_size[1]:
                                    continue
                                tbb_image = self.get_bb_image(image, tbox2d_inside)
                                thist = self.get_hist(tbb_image, hist_mask[cls])
                                tsim = self.get_similality(thist, p['hist'])
                                if tsim>similarity:
                                    similarity = tsim
                                    pos = tpos
                                    bb_image = tbb_image
                                    updated = True
                            if (not updated) or count>8:
                                break
                            else:
                                count += 1
                        thresh = 0.85 if cls=='Pedestrian' else 0.9
                        if similarity>thresh:
                            interp = 1
                            if 'intep' in p.keys():
                                interp = p['interp'] + 1
                            if interp<=1:
                                ibb = [box2d_inside[0]+pos[0], box2d_inside[1]+pos[1], box2d_inside[2]+pos[0], box2d_inside[3]+pos[1]]
                                boxes.append({'box2d': ibb, 'score': p['score'], 'interp': interp})

            # prepare image inside each bounding box
            for box in boxes:
                # just for evaluation with GT bboxes
                if 'score' not in box.keys():
                    box['score'] = -1
                bb = box['box2d']
                im = self.get_bb_image(image, bb)
                box['image'] = im
                im = self.smooth_image(im)
                hist = self.get_hist(im, hist_mask[cls])
                box['hist'] = hist

            # match objects in the previous frame and the current frame and assign IDs
            box_map, cost = self.hungarian_match(adjusted_preds, boxes, cls)
            self.total_cost += cost
            prev_ids = list(map(lambda p: p['id'], adjusted_preds))
            next_ids = [prev_ids[box_map.index(i)] if i in box_map else -1 for i in range(len(boxes))]
            for i in range(len(next_ids)):
                if(next_ids[i]==-1):
                    next_ids[i] = self.last_id + 1
                    self.last_id += 1

            # update object information (speed of position, scaling, etc.) to keep in the tracker
            for i in range(len(boxes)):
                if next_ids[i] in prev_ids:
                    prev_box2d = adjusted_preds[prev_ids.index(next_ids[i])]['box2d']
                    box2d = pred[cls][i]['box2d']

                    # calculate motion vector of each object
                    prev_cnt = [(prev_box2d[0]+prev_box2d[2])//2, (prev_box2d[1]+prev_box2d[3])//2]
                    cnt = [(box2d[0]+box2d[2])//2, (box2d[1]+box2d[3])//2]
                    mv = [cnt[0]-prev_cnt[0], cnt[1]-prev_cnt[1]]
                    scale = [1, 1]
                else:
                    mv = [0, 0]
                    scale = [1, 1]
                bb = pred[cls][i]['box2d']
                pp = {'box2d': pred[cls][i]['box2d'], 'score': pred[cls][i]['score'], 'id': next_ids[i], 'mv': mv, 'scale': scale, 'occlusion': 0, 'image': pred[cls][i]['image'], 'hist': pred[cls][i]['hist']}
                if 'interp' in pred[cls][i].keys():
                    pp['interp'] = pred[cls][i]['interp']
                pred[cls][i] = pp

            # generate next prediction data
            for i in range(len(box_map)):
                # discard too old occluded objects kept in the tracker
                if box_map[i]==-1 and adjusted_preds[i]['occlusion']<self.max_occ_frames:
                    bb = adjusted_preds[i]['box2d']
                    pp = {'box2d': bb, 'score': adjusted_preds[i]['score'], 'id': adjusted_preds[i]['id'], 'mv': adjusted_preds[i]['mv'], 'scale': adjusted_preds[i]['scale'], 'occlusion': adjusted_preds[i]['occlusion']+1, 'image': adjusted_preds[i]['image'], 'hist': adjusted_preds[i]['hist']}
                    if 'interp' in adjusted_preds[i]:
                        pp['interp'] = adjusted_preds[i]['interp']
                    pred[cls].append(pp)

            # filter out matching with too low similarity
            tpred = []
            for tp in pred[cls]:
                if tp['id'] in map(lambda p: p['id'], last_preds):
                    pp = list(filter(lambda p: p['id']==tp['id'], last_preds))[0]
                    hist1 = tp['hist']
                    hist2 = pp['hist']
                    similarity = self.get_similality(hist1, hist2)
                    if similarity>0.2:
                        tpred.append(tp)
                    else:
                        pp['occlusion'] = 1
                        tpred.append(pp)
                        tp['id'] = self.last_id + 1
                        self.last_id += 1
                        tpred.append(tp)
                else:
                    tpred.append(tp)
            pred[cls] = tpred

        # keep object prediction information in the tracker
        self.predictions.append(pred)

        ret = copy.deepcopy(pred)
        for cls in ret.keys():
            tmp = []
            for box in ret[cls]:
                # return prediction data excluding occluded objects
                if box['occlusion']==0:
                    tmp.append({'box2d': box['box2d'], 'id': box['id']})
            ret[cls] = tmp
        return ret


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--input_pred", type=str, dest="input_pred_path", required=True, help="input prediction directory path")
    parser.add_argument("--input_video", type=str, dest="input_video_path", required=True, help="input video directory path")
    parser.add_argument("-o", "--output", type=str, dest="output_path", required=True, help="output file path")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="output file path")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="verbose")
    parser.add_argument("-p", "--process", dest="nproc", type=int, default=1, help='The max number of process')
    args = parser.parse_args()

    video_total = {'Car': 0, 'Pedestrian': 0}
    video_error = {'Car': 0, 'Pedestrian': 0}

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (255, 128, 128), (128, 255, 0), (0, 255, 128), (128, 255, 128), (128, 0, 255), (0, 128, 255),
        (128, 128, 255), (128, 128, 128), (0, 0, 0), (255, 255, 255),
    ]

    if args.debug:
        if not os.path.exists('debug'):
            os.mkdir('debug')

    def evaluate_video(pred):
        with open(pred) as f:
            ground_truths = json.load(f)
        ground_truths = ground_truths['sequence']
        ground_truths = list(map(lambda x: {'Car': x['Car'] if 'Car' in x.keys() else [], 'Pedestrian': x['Pedestrian'] if 'Pedestrian' in x.keys() else []}, ground_truths))
        video_name = os.path.basename(pred)
        video = os.path.join(args.input_video_path, video_name.split('.')[0]+'.mp4')
        video = cv2.VideoCapture(video)
        tracker = Tracker((1936, 1216))
        total = {'Car': 0, 'Pedestrian': 0}
        sw = {'Car': 0, 'Pedestrian': 0}
        tp = {'Car': 0, 'Pedestrian': 0}
        max_time = 0
        if args.debug:
            if os.path.exists(os.path.join('debug', video_name.split('.')[0])):
                shutil.rmtree(os.path.join('debug', video_name.split('.')[0]))
            os.mkdir(os.path.join('debug', video_name.split('.')[0]))
        for frame in range(len(ground_truths)):
            if frame%100==0 and args.verbose:
                print(f'"{video_name}" Frame {frame+1}: ', end='')
            _, image = video.read()
            ground_truth = ground_truths[frame]
            prediction = copy.deepcopy(ground_truth)
            t1 = time.time()
            prediction = tracker.assign_ids(prediction, image)
            if frame==0:
                prev_id_map = {'Car': {}, 'Pedestrian': {}}
                for cls, gt in ground_truth.items():
                    for g in gt:
                        gt_id = g['id']
                        gt_bb = g['box2d']
                        if (gt_bb[2]-gt_bb[0]+1)*(gt_bb[3]-gt_bb[1]+1)<1024:
                            continue
                        m_id = -1
                        for p in prediction[cls]:
                            p_id = p['id']
                            p_bb = p['box2d']
                            if gt_bb==p_bb:
                                m_id = p_id
                                break
                        prev_id_map[cls][gt_id] = m_id
                prev_image = image
            else:
                if args.debug:
                    debug_image1 = prev_image.copy()
                    debug_image2 = image.copy()
                    debug_idx = 0
                id_map = {'Car': {}, 'Pedestrian': {}}
                for cls, gt in ground_truth.items():
                    bm = 0
                    for g in gt:
                        gt_id = g['id']
                        gt_bb = g['box2d']
                        if (gt_bb[2]-gt_bb[0]+1)*(gt_bb[3]-gt_bb[1]+1)<1024:
                            continue
                        total[cls] += 1
                        m_id = -1
                        for p in prediction[cls]:
                            p_id = p['id']
                            p_bb = p['box2d']
                            if gt_bb==p_bb:
                                m_id = p_id
                                bm += 1
                                break
                        id_map[cls][gt_id] = m_id
                for cls, gt in ground_truth.items():
                    for g in gt:
                        gt_id = g['id']
                        gt_bb = g['box2d']
                        if (gt_bb[2]-gt_bb[0]+1)*(gt_bb[3]-gt_bb[1]+1)<1024:
                            continue
                        if gt_id in prev_id_map[cls].keys():
                            prev_m_id = prev_id_map[cls][gt_id]
                            if gt_id in id_map[cls].keys():
                                if prev_m_id!=id_map[cls][gt_id]:
                                    if args.debug:
                                        debug_bb1 = list(filter(lambda p: p['id']==prev_m_id, tracker.predictions[-2][cls]))
                                        debug_bb2 = list(filter(lambda p: p['id']==prev_m_id, tracker.predictions[-1][cls]))
                                        if len(debug_bb1)>0 and len(debug_bb2)>0:
                                            debug_bb1 = debug_bb1[0]['box2d']
                                            debug_bb2 = debug_bb2[0]['box2d']
                                            debug_image1 = cv2.rectangle(debug_image1, (debug_bb1[0], debug_bb1[1]), (debug_bb1[2], debug_bb1[3]), colors[debug_idx], 3)
                                            debug_image2 = cv2.rectangle(debug_image2, (debug_bb2[0], debug_bb2[1]), (debug_bb2[2], debug_bb2[3]), colors[debug_idx], 3)
                                            debug_idx += 1
                                    sw[cls] += 1
                                else:
                                    tp[cls] += 1
                for k, v in id_map.items():
                    prev_id_map[k] = v
                if args.debug:
                    debug_image = np.concatenate([debug_image1, debug_image2], axis=1)
                    cv2.imwrite(os.path.join('debug', video_name.split('.')[0], f'{frame}.png'), debug_image)
                prev_image = image

            t2 = time.time()
            max_time = max(max_time, t2-t1)
            if frame%100==0 and args.verbose:
                print(f'#Car={len(ground_truth["Car"])}, #Pedestrian={len(ground_truth["Pedestrian"])}, ', end='')
                print(f'Time={t2-t1:.8f}({max_time:.8f}@max), Cost={tracker.total_cost}')
        print(f'Overall ({video_name})')
        record = {'Name': video_name}
        for cls in sw.keys():
            record[cls] = sw[cls]/total[cls]
            print(f'    {cls}: total={total[cls]}, sw={sw[cls]}, tp={tp[cls]}, err={sw[cls]/total[cls]:.8f}')
        record['Avg'] = (sw['Car']/total['Car']+sw['Pedestrian']/total['Pedestrian']) / 2
        print(f'    All: err={(sw["Car"]/total["Car"]+sw["Pedestrian"]/total["Pedestrian"])/2:.8f}')
        return sw, total, record

    executor = ProcessPoolExecutor(max_workers=args.nproc)
    q = []
    records = []
    input_files = sorted(glob(os.path.join(args.input_pred_path, '*')))
    res = executor.map(evaluate_video, input_files)
    for r in res:
        sw, total, record = r
        for cls in sw.keys():
            video_total[cls] += 1
            video_error[cls] += sw[cls]/total[cls]
        records.append(record)

    print(f'Complete Result:')
    print(f'    Car: {video_error["Car"]/video_total["Car"]:.8f}')
    print(f'    Pedestrian: {video_error["Pedestrian"]/video_total["Pedestrian"]:.8f}')
    print(f'    All: err={(video_error["Car"]/video_total["Car"]+video_error["Pedestrian"]/video_total["Pedestrian"])/2:.8f}')
    print()
    print('Short Log:')
    for record in records:
        print(f'{record["Name"]}: {record["Car"]:.5f}(Car), {record["Pedestrian"]:.5f}(Pedestrian), {record["Avg"]:.5f}(Avg)')
    print(f'Average      : {video_error["Car"]/video_total["Car"]:.5f}(Car), {video_error["Pedestrian"]/video_total["Pedestrian"]:.5f}(Pedestrian), {(video_error["Car"]/video_total["Car"]+video_error["Pedestrian"]/video_total["Pedestrian"])/2:.5f}(Avg)')

