import cv2
import os
import numpy as np
import copy
import json
import time
from glob import glob
from argparse import ArgumentParser


class Tracker:
    def __init__(self, image_size):
        self.predictions = [{'Car': [], 'Pedestrian': []}]; # {'id': id, 'box2d': [x1, y1, x2, y2], 'mv': [vx, vy], 'scale': [sx, sy], 'occlusion': number_of_occlusions}
        self.image_size = image_size
        self.max_occ_frames = 3
        self.max_occ = {'Car': 5, 'Pedestrian': 4}
        self.max_frame_in = 2
        self.frame_out_thresh = 0.5
        self.cost_thresh1 = {'Car': 0.7, 'Pedestrian': 1.3}
        self.cost_thresh2 = {'Car': 1.2, 'Pedestrian': 1.7}
        self.cost_weight = {'Car': [1.0, 1.3], 'Pedestrian': [1.0, 1.3]}
        self.box_area_thresh = 1024
        self.total_cost = 0
        self.cls_to_track = ["Car", "Pedestrian"]
        self.init_cls_pred = [{'Car': [], 'Pedestrian': []}]


    def calculate_cost(self, box1, box2, cls='Car'):
        w1, h1 = box1[2]-box1[0]+1, box1[3]-box1[1]+1
        w2, h2 = box2[2]-box2[0]+1, box2[3]-box2[1]+1
        cnt1 = [box1[0]+w1/2, box1[1]+h1/2]
        cnt2 = [box2[0]+w2/2, box2[1]+h2/2]
        alpha = abs(cnt1[0]-cnt2[0])/(w1+w2) + abs(cnt1[1]-cnt2[1])/(h1+h2)
        if cls=='Car':
            beta = (w1+w2)/(2*np.sqrt(w1*w2)) + (h1+h2)/(2*np.sqrt(h1*h2))
        else:
            beta = (w1+w2)/(2*np.sqrt(w1*w2)) * (h1+h2)/(2*np.sqrt(h1*h2))
        cost = pow(alpha, self.cost_weight[cls][0]) * pow(beta, self.cost_weight[cls][1])
        return cost


    def match(self, preds1, preds2, cls='Car'):
        n1 = len(preds1)
        n2 = len(preds2)
        match_costs = [[0]*n2 for _ in range(n1)]
        cands = [[] for _ in range(n1)]
        for i in range(n1):
            for j in range(n2):
                match_costs[i][j] = self.calculate_cost(preds1[i]['box2d'], preds2[j]['box2d'], cls)
                cands[i].append(j)
        for i in range(n1):
            tmp = list(filter(lambda x: match_costs[i][x]<=self.cost_thresh1[cls], cands[i]))
            if len(tmp)>=2:
                cands[i] = tmp
                cands[i].sort(key=lambda x: match_costs[i][x])
            else:
                tmp = list(filter(lambda x: match_costs[i][x]<=self.cost_thresh2[cls], cands[i]))
                cands[i] = tmp
                cands[i].sort(key=lambda x: match_costs[i][x])
            cands[i] = cands[i][:max(1, 128//(n1+n2))]
        limit_occ = min(n1, self.max_occ[cls])
        limit_frame_in = min(n2, self.max_frame_in)
        best_box_map = []
        min_cost = 1e16

        count = 0
        # @profile
        def rec_match(rem_match, idx=0, box_map=[], curr_cost=0):
            nonlocal min_cost
            if curr_cost>=min_cost:
                return
            if rem_match==0:
                if curr_cost<min_cost:
                    min_cost = curr_cost
                    nonlocal best_box_map
                    best_box_map = box_map
                return
            if rem_match>=n1-idx:
                return
            rec_match(rem_match, idx+1, box_map+[-1], curr_cost)
            for i in cands[idx]:
                if i in box_map:
                    continue
                rec_match(rem_match-1, idx+1, box_map+[i], curr_cost+match_costs[idx][i])

        for n_occ in range(limit_occ):
            n_match = n1 - n_occ
            if n_match>n2:
                continue
            for n_frame_in in range(n2-n_match):
                rec_match(n_match, curr_cost=n_occ+n_frame_in)

        self.total_cost += min_cost

        return best_box_map



    def assign_ids(self, pred): # {'Car': [{'box2d': [x1, y1, x2, y2]}], 'Pedestrian': [{'box2d': [x1, y1, x2, y2]}]}
        pred = copy.deepcopy(pred)

        for cls, boxes in pred.items():
            # Issue if the detection model didn't detect the Classe in previous frame, the tracker failed.
            if cls not in self.predictions[-1]:
                last_preds = self.init_cls_pred[-1][cls]                # Empty prediction
            else:
                last_preds = self.predictions[-1][cls]

            adjusted_preds = []
            n_frame_out = 0
            for p in last_preds:
                print("p in last_preds: {}".format(p))
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
                # mv = [mv[0]*scale[0], mv[1]*scale[1]]
                cnt = [(box2d[2]+box2d[0])/2, (box2d[3]+box2d[1])/2]
                print(cnt)
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
                adjusted_preds.append({'id': p['id'], 'box2d': box2d_inside, 'mv': p['mv'], 'scale': p['scale'], 'occlusion': p['occlusion']})
            box_map = self.match(adjusted_preds, boxes, cls)
            prev_ids = list(map(lambda p: p['id'], adjusted_preds))
            next_ids = [prev_ids[box_map.index(i)] if i in box_map else -1 for i in range(len(boxes))]
            next_id = 0
            for i in range(len(next_ids)):
                if(next_ids[i]==-1):
                    while True:
                        if next_id not in next_ids:
                            next_ids[i] = next_id
                            next_id += 1
                            break
                        next_id += 1
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
                    print("New object")
                    mv = [0, 0]
                    scale = [1, 1]
                pred[cls][i] = {'box2d': pred[cls][i]['box2d'], 'id': next_ids[i], 'mv': mv, 'scale': scale, 'occlusion': 0}
            for i in range(len(box_map)):
                if box_map[i]==-1 and adjusted_preds[i]['occlusion']<self.max_occ_frames:
                    pred[cls].append({'box2d': adjusted_preds[i]['box2d'], 'id': adjusted_preds[i]['id'], 'mv': adjusted_preds[i]['mv'], 'scale': adjusted_preds[i]['scale'], 'occlusion': adjusted_preds[i]['occlusion']+1})
        
        # Save prediction for future steps:
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
    parser.add_argument("-i", "--input", type=str, dest="input_path", required=True, help="input directory path")
    parser.add_argument("-o", "--output", type=str, dest="output_path", required=True, help="output file path")
    args = parser.parse_args()


    for video in glob(os.path.join(args.input_path, '*')):
        max_time = 0
        with open(video) as f:
            ground_truths = json.load(f)
        ground_truths = ground_truths['sequence']
        ground_truths = list(map(lambda x: {'Car': x['Car'] if 'Car' in x.keys() else [], 'Pedestrian': x['Pedestrian'] if 'Pedestrian' in x.keys() else []}, ground_truths))
        video_name = os.path.basename(video)
        tracker = Tracker((1936, 1216))
        total = {'Car': 0, 'Pedestrian': 0}
        sw = {'Car': 0, 'Pedestrian': 0}
        tp = {'Car': 0, 'Pedestrian': 0}
        for frame in range(len(ground_truths)):
            print(f'Frame #{frame+1}')
            ground_truth = ground_truths[frame]
            prediction = copy.deepcopy(ground_truth)
            t1 = time.time()
            prediction = tracker.assign_ids(prediction)
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
            else:
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
                                    sw[cls] += 1
                                else:
                                    tp[cls] += 1
                for k, v in id_map.items():
                    prev_id_map[k] = v

            t2 = time.time()
            max_time = max(max_time, t2-t1)
            print(f'    #Boxes: Car={len(ground_truth["Car"])}, Pedestrian={len(ground_truth["Pedestrian"])}')
            print(f'    Execution time: total={t2-t1:.8f}, max={max_time:.8f}')
            print('    Total cost: ', tracker.total_cost)
        print(f'Overall ({video})')
        for cls in sw.keys():
            print(f'    {cls}: total={total[cls]}, sw={sw[cls]}, tp={tp[cls]}, err={sw[cls]/total[cls]:.8f}')
        print(f'    All: err={(sw["Car"]+sw["Pedestrian"])/(total["Car"]+total["Pedestrian"]):.8f}')
        # exit()
