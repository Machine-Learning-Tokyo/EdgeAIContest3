import cv2
import os
import numpy as np
import copy
import json
import time
import shutil
from glob import glob
from argparse import ArgumentParser
from statistics import mean


class Tracker:
    def __init__(self, image_size, max_patterns=10000000, max_time=0.1):
        self.predictions = [{'Car': [], 'Pedestrian': []}]; # {'id': id, 'box2d': [x1, y1, x2, y2], 'mv': [vx, vy], 'scale': [sx, sy], 'occlusion': number_of_occlusions, 'image': image}
        self.image_size = image_size
        self.max_occ_frames = 20
        self.frame_out_thresh = 0.2
        self.box_area_thresh = 1024
        self.max_patterns = max_patterns
        self.max_time = max_time
        self.max_frame_in = {'Car': 4, 'Pedestrian': 5}
        self.cost_thresh1 = {'Car': 0.35, 'Pedestrian': 0.83}
        self.cost_thresh2 = {'Car': 0.71, 'Pedestrian': 1.44}
        self.cost_weight = {'Car': [0.5, 1.21], 'Pedestrian': [0.2, 1.16]}
        self.sim_weight = {'Car': 1.47, 'Pedestrian': 1.75}
        self.occ_weight = {'Car': 0.84, 'Pedestrian': 1.46}
        self.last_id = -1
        self.total_cost = 0

        self.init_cls_pred = [{'Car': [], 'Pedestrian': []}]


    def calculate_cost(self, box1, box2, hist1, hist2, cls='Car'):
        w1, h1 = box1[2]-box1[0]+1, box1[3]-box1[1]+1
        w2, h2 = box2[2]-box2[0]+1, box2[3]-box2[1]+1
        hist_score = [cv2.compareHist(hist1[c], hist2[c], cv2.HISTCMP_CORREL) for c in range(3)]
        # hist_score = mean(hist_score)
        hist_score = min(hist_score)
        cnt1 = [box1[0]+w1/2, box1[1]+h1/2]
        cnt2 = [box2[0]+w2/2, box2[1]+h2/2]
        alpha = abs(cnt1[0]-cnt2[0])/(w1+w2) + abs(cnt1[1]-cnt2[1])/(h1+h2)
        beta = (w1+w2)/(2*np.sqrt(w1*w2)) * (h1+h2)/(2*np.sqrt(h1*h2))
        cost = pow(alpha, self.cost_weight[cls][0]) * pow(beta, self.cost_weight[cls][1]) * pow(2, (0.5-hist_score)*self.sim_weight[cls])
        return cost


    def match(self, preds1, preds2, cls='Car'):
        n1 = len(preds1)
        n2 = len(preds2)
        match_costs = [[0]*n2 for _ in range(n1)]
        cands = [[] for _ in range(n1)]
        all_cands = [[] for _ in range(n1)]
        hist1s = [[cv2.calcHist([cv2.resize(preds1[i]['image'], (64, 64), interpolation=cv2.INTER_CUBIC)], [c], None, [64], [0, 256]) for c in range(3)] for i in range(n1)]
        hist2s = [[cv2.calcHist([cv2.resize(preds2[i]['image'], (64, 64), interpolation=cv2.INTER_CUBIC)], [c], None, [64], [0, 256]) for c in range(3)] for i in range(n2)]
        for i in range(n1):
            for j in range(n2):
                match_costs[i][j] = self.calculate_cost(preds1[i]['box2d'], preds2[j]['box2d'], hist1s[i], hist2s[j], cls)
                cands[i].append(j)
                all_cands[i].append(j)
        for i in range(n1):
            all_cands[i].sort(key=lambda x: match_costs[i][x])
            tmp = list(filter(lambda x: match_costs[i][x]<=self.cost_thresh1[cls], cands[i]))
            if len(tmp)>=3:
                cands[i] = tmp
                cands[i].sort(key=lambda x: match_costs[i][x])
            else:
                tmp = list(filter(lambda x: match_costs[i][x]<=self.cost_thresh2[cls], cands[i]))
                if len(tmp)>=1:
                    cands[i] = tmp
                    cands[i].sort(key=lambda x: match_costs[i][x])
                else:
                    cands[i].sort(key=lambda x: match_costs[i][x])
            cands[i] = cands[i][:max(1, 150//(n1+n2))]
        best_box_map = []
        min_cost = 1e16

        # find at least one candidate to avoid no matching
        found1 = 0
        def rec_match_find1(rem_match, idx=0, box_map=[], curr_cost=0):
            nonlocal found1
            if found1>100:
                return
            if rem_match==0:
                found1 += 1
                nonlocal min_cost
                if curr_cost<min_cost:
                    min_cost = curr_cost
                    nonlocal best_box_map
                    best_box_map = box_map + [-1]*max(0, n1-len(box_map))
                return
            cnt = 0
            for i in all_cands[idx]:
                if i in box_map:
                    continue
                rec_match_find1(rem_match-1, idx+1, box_map+[i], curr_cost+match_costs[idx][i])

        count = 0
        start_time = time.time()
        time_over = False
        def rec_match(rem_match, idx=0, box_map=[], curr_cost=0):
            nonlocal count
            nonlocal time_over
            count += 1
            if count>self.max_patterns or time_over:
                return
            if count%10000==0:
                current_time = time.time()
                if current_time-start_time>self.max_time:
                    time_over = True
            nonlocal min_cost
            if curr_cost>=min_cost:
                return
            if rem_match==0:
                if curr_cost<min_cost:
                    min_cost = curr_cost
                    nonlocal best_box_map
                    best_box_map = box_map + [-1]*max(0, n1-len(box_map))
                return
            if rem_match>=n1-idx:
                return
            rec_match(rem_match, idx+1, box_map+[-1], curr_cost)
            for i in cands[idx]:
                if i in box_map:
                    continue
                rec_match(rem_match-1, idx+1, box_map+[i], curr_cost+match_costs[idx][i])

        rec_match_find1(min(n1, n2), curr_cost=(n1-min(n1, n2))*self.occ_weight[cls]+(n2-min(n1, n2)))
        for n_occ in range(max(n1-n2, 0), max(n1-n2+self.max_frame_in[cls]+1, min(n2, self.max_frame_in[cls]))):
            n_match = n1 - n_occ
            if n_match>n2:
                continue
            for n_frame_in in range(n2-n_match+1):
                rec_match(n_match, curr_cost=n_occ*self.occ_weight[cls]+n_frame_in)

        if n1>0 or n2>0:
            self.total_cost += min_cost

        return best_box_map



    def assign_ids(self, pred, image): # {'Car': [{'box2d': [x1, y1, x2, y2]}], 'Pedestrian': [{'box2d': [x1, y1, x2, y2]}]}
        pred = copy.deepcopy(pred)
        for cls, boxes in pred.items():
            # Ben quick
            if cls not in self.predictions[-1]:
                last_preds = self.init_cls_pred[-1][cls]
            else:
                last_preds = self.predictions[-1][cls]

            adjusted_preds = []
            n_frame_out = 0
            for box in boxes:
                bb = box['box2d']
                bb[0] = max(0, bb[0])
                bb[1] = max(0, bb[1])
                bb[2] = min(self.image_size[0]-1, bb[2])
                bb[3] = min(self.image_size[1]-1, bb[3])
                bb = [int(min(bb[0], bb[2])), int(min(bb[1], bb[3])), int(max(bb[0], bb[2])), int(max(bb[1], bb[3]))]   # TypeError: slice indices must be integers or None or have an __index__ method
                box['image'] = image[bb[1]:bb[3]+1, bb[0]:bb[2]+1, :]
            for p in last_preds:
                box2d = p['box2d']
                mv = p['mv']
                if len(self.predictions)>=2:
                    # Same here
                    if cls not in self.predictions[-2]:
                        last2_preds = self.init_cls_pred[-1][cls]
                    else:
                        last2_preds = self.predictions[-2][cls]

                    if p['id'] in map(lambda p2: p2['id'], last2_preds):
                        p2 = list(filter(lambda p2: p2['id']==p['id'], last2_preds))[0]
                        mv2 = p2['mv']
                        a = [mv[0]-mv2[0], mv[1]-mv2[1]]
                        if abs(mv[0])>abs(a[0])*2 and abs(mv[1])>abs(a[1])*2:
                            mv = [mv[0]+a[0], mv[1]+a[1]]
                scale = p['scale']
                cnt = [(box2d[2]+box2d[0])/2, (box2d[3]+box2d[1])/2]
                w = box2d[2]-box2d[0]+1
                h = box2d[3]-box2d[1]+1
                sw = w * scale[0]
                sh = h * scale[1]
                x1 = int(cnt[0] - sw/2 + mv[0])
                x2 = int(cnt[0] + sw/2 + mv[0])
                y1 = int(cnt[1] - sh/2 + mv[1])
                y2 = int(cnt[1] + sh/2 + mv[1])
                box2d = [max(0, x1), max(0, y1), min(self.image_size[0]-1, x2), min(self.image_size[1]-1, y2)]
                box2d = [min(box2d[0], box2d[2]), min(box2d[1], box2d[3]), max(box2d[0], box2d[2]), max(box2d[1], box2d[3])]
                area = (box2d[2]-box2d[0]+1) * (box2d[3]-box2d[1]+1)
                if area<self.box_area_thresh:
                    continue
                box2d_inside = [max(0, box2d[0]), max(0, box2d[1]), min(self.image_size[0]-1, box2d[2]), min(self.image_size[1]-1, box2d[3])]
                area_inside = (box2d_inside[2]-box2d_inside[0]+1) * (box2d_inside[3]-box2d_inside[1]+1)
                if area_inside <=area*self.frame_out_thresh:
                    n_frame_out += 1
                    continue
                adjusted_preds.append({'id': p['id'], 'box2d': box2d_inside, 'mv': p['mv'], 'scale': p['scale'], 'occlusion': p['occlusion'], 'image': p['image']})
            box_map = self.match(adjusted_preds, boxes, cls)
            prev_ids = list(map(lambda p: p['id'], adjusted_preds))
            next_ids = [prev_ids[box_map.index(i)] if i in box_map else -1 for i in range(len(boxes))]
            for i in range(len(next_ids)):
                if(next_ids[i]==-1):
                    next_ids[i] = self.last_id + 1
                    self.last_id += 1
            for i in range(len(boxes)):
                if next_ids[i] in prev_ids:
                    prev_box2d = adjusted_preds[prev_ids.index(next_ids[i])]['box2d']
                    box2d = pred[cls][i]['box2d']
                    prev_cnt = [(prev_box2d[0]+prev_box2d[2])//2, (prev_box2d[1]+prev_box2d[3])//2]
                    cnt = [(box2d[0]+box2d[2])//2, (box2d[1]+box2d[3])//2]
                    mv = [cnt[0]-prev_cnt[0], cnt[1]-prev_cnt[1]]
                    sx = (box2d[2]-box2d[0]+1) / (prev_box2d[2]-prev_box2d[0]+1)
                    sy = (box2d[3]-box2d[1]+1) / (prev_box2d[3]-prev_box2d[1]+1)
                    scale = [sx, sy]
                else:
                    mv = [0, 0]
                    scale = [1, 1]
                bb = pred[cls][i]['box2d']
                pred[cls][i] = {'box2d': pred[cls][i]['box2d'], 'id': next_ids[i], 'mv': mv, 'scale': scale, 'occlusion': 0, 'image': image[int(bb[1]):int(bb[3])+1, int(bb[0]):int(bb[2])+1, :]}
            for i in range(len(box_map)):
                if box_map[i]==-1 and adjusted_preds[i]['occlusion']<self.max_occ_frames:
                    bb = adjusted_preds[i]['box2d']
                    pred[cls].append({'box2d': bb, 'id': adjusted_preds[i]['id'], 'mv': adjusted_preds[i]['mv'], 'scale': adjusted_preds[i]['scale'], 'occlusion': adjusted_preds[i]['occlusion']+1, 'image': image[int(bb[1]):int(bb[3])+1, int(bb[0]):int(bb[2])+1, :]})
        self.predictions.append(pred)
        ret = copy.deepcopy(pred)
        for cls in ret.keys():
            tmp = []
            for box in ret[cls]:
                if box['occlusion']==0:
                    tmp.append({'box2d': box['box2d'], 'id': box['id']})
            ret[cls] = tmp
        return ret


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-i", "--input_pred", type=str, dest="input_pred_path", required=True, help="input prediction directory path")
    parser.add_argument("-v", "--input_video", type=str, dest="input_video_path", required=True, help="input video directory path")
    parser.add_argument("-o", "--output", type=str, dest="output_path", required=True, help="output file path")
    args = parser.parse_args()

    video_total = {'Car': 0, 'Pedestrian': 0}
    video_error = {'Car': 0, 'Pedestrian': 0}

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (255, 128, 128), (128, 255, 0), (0, 255, 128), (128, 255, 128), (128, 0, 255), (0, 128, 255),
        (128, 128, 255), (128, 128, 128), (0, 0, 0), (255, 255, 255),
    ]

    for nv, pred in enumerate(sorted(glob(os.path.join(args.input_pred_path, '*')))):
        max_time = 0
        # if nv<24:
            # continue
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
        if os.path.exists(os.path.join('debug', video_name.split('.')[0])):
            shutil.rmtree(os.path.join('debug', video_name.split('.')[0]))
        os.mkdir(os.path.join('debug', video_name.split('.')[0]))
        for frame in range(len(ground_truths)):
            if frame%100==0:
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
                debug_image = np.concatenate([debug_image1, debug_image2], axis=1)
                cv2.imwrite(os.path.join('debug', video_name.split('.')[0], f'{frame}.png'), debug_image)
                prev_image = image

            t2 = time.time()
            max_time = max(max_time, t2-t1)
            if frame%100==0:
                print(f'#Car={len(ground_truth["Car"])}, #Pedestrian={len(ground_truth["Pedestrian"])}, ', end='')
                print(f'Time={t2-t1:.8f}({max_time:.8f}@max), Cost={tracker.total_cost}')
        print(f'Overall ({video_name})')
        for cls in sw.keys():
            video_total[cls] += 1
            video_error[cls] += sw[cls]/total[cls]
            print(f'    {cls}: total={total[cls]}, sw={sw[cls]}, tp={tp[cls]}, err={sw[cls]/total[cls]:.8f}')
        print(f'    All: err={(sw["Car"]/total["Car"]+sw["Pedestrian"]/total["Pedestrian"])/2:.8f}')
    print(f'complete Result')
    print(f'    Car: {video_error["Car"]/video_total["Car"]:.8f}')
    print(f'    Pedestrian: {video_error["Pedestrian"]/video_total["Pedestrian"]:.8f}')
    print(f'    All: err={(video_error["Car"]/video_total["Car"]+video_error["Pedestrian"]/video_total["Pedestrian"])/2:.8f}')
