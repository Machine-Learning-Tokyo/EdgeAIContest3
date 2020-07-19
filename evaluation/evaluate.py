# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:19:44 2020

@author: n.aoi
"""

import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser

class Correspondence():
    def __init__(self, threshold):
        self.cumu_matches = {}
        self.temp_misses = {}
        self.threshold = threshold
    

    def count_fp_fn_idsw(self, true, pred): 
        results = {}
        matches = {}

        intersection_categories = set(true).intersection(set(pred))
        for intersection_category in intersection_categories:
            gt = true[intersection_category]
            pr = pred[intersection_category]
            matches[intersection_category] = {}
            results[intersection_category] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            results[intersection_category]['GT'] = len(gt)
            if intersection_category in self.cumu_matches:
                cumu_matches = self.cumu_matches[intersection_category]
                                                
                m = self.find_temp_match(gt, pr, cumu_matches, self.threshold)
                matches[intersection_category].update(m)

                gt_unmatched = list(filter(lambda x: x['id'] not in matches[intersection_category] and x['id'] in cumu_matches, gt))
                pr_unmatched = list(filter(lambda x: x['id'] not in matches[intersection_category].values(), pr))
                m = self.find_match(gt_unmatched, pr_unmatched, self.threshold)
                results[intersection_category]['IDSW'] += len(m)
                matches[intersection_category].update(m)

                new_objects = list(filter(lambda x: x['id'] not in cumu_matches, gt))
                pr_objects = list(filter(lambda x: x['id'] not in matches[intersection_category].values(), pr))
                m = self.find_match(new_objects, pr_objects, self.threshold)
                matches[intersection_category].update(m)
                    
                results[intersection_category]['FN'] += sum(map(lambda x: x['id'] not in matches[intersection_category], gt))
                results[intersection_category]['FP'] += sum(map(lambda x: x['id'] not in matches[intersection_category].values(), pr))
            else:                                
                m = self.find_match(gt, pr, self.threshold)
                matches[intersection_category].update(m)
                    
                results[intersection_category]['FN'] += sum(map(lambda x: x['id'] not in matches[intersection_category], gt))
                results[intersection_category]['FP'] += sum(map(lambda x: x['id'] not in matches[intersection_category].values(), pr))
                
        
        pred_true_difference_categories = set(pred).difference(set(true))
        for pred_true_difference_category in pred_true_difference_categories:
            results[pred_true_difference_category] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            pr = pred[pred_true_difference_category]
            results[pred_true_difference_category]['FP'] += len(pr)
            
        true_pred_difference_categories = set(true).difference(set(pred))
        for true_pred_difference_category in true_pred_difference_categories:
            results[true_pred_difference_category] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            gt = true[true_pred_difference_category]
            results[true_pred_difference_category]['GT'] += len(gt)
            results[true_pred_difference_category]['FN'] += len(gt)
        
        
        ## update cumulative matches
        for c, cumu_match in self.cumu_matches.items():
            if c in matches:
                self.cumu_matches[c].update(matches[c])
        new_categories = set(matches).difference(set(self.cumu_matches))
        for new_category in new_categories:
            self.cumu_matches[new_category] = matches[new_category]
        
        ## update misses(fn, fp, idsw)
        self.temp_misses = results
    

    def find_match(self, gt_objects, pr_objects, threshold):
        result = {}
        if len(gt_objects) and len(pr_objects):
            mat = []
            for gt_object in gt_objects:
                mat.append([compute_iou_bb(pr_object['box2d'], gt_object['box2d']) for pr_object in pr_objects])
            profit_array = np.array(mat)
            cost_array = 1 - np.array(mat)
            
            row_ind, col_ind = linear_sum_assignment(cost_array)
            matches = np.array((row_ind, col_ind)).T
            result.update({gt_objects[i]['id']: pr_objects[j]['id'] for i, j in matches if profit_array[i][j] >= threshold})
        
        return result


    def find_temp_match(self, gt_objects, pr_objects, matches, threshold):
        result = {}
        for g_id, p_id in matches.items():
            g_object = list(filter(lambda x: x['id'] == g_id, gt_objects))
            p_object = list(filter(lambda x: x['id'] == p_id, pr_objects))
            if len(g_object) == 1 and len(p_object) == 1:
                iou = compute_iou_bb(p_object[0]['box2d'], g_object[0]['box2d'])
                if iou >= threshold:
                    result[g_id] = p_id

        return result


def compute_iou_bb(pred_bb, true_bb):
    pred_area = (pred_bb[2] - pred_bb[0])*(pred_bb[3] - pred_bb[1])
    true_area = (true_bb[2] - true_bb[0])*(true_bb[3] - true_bb[1])
    intersection_x = max(min(pred_bb[2], true_bb[2]) - max(pred_bb[0], true_bb[0]), 0)
    intersection_y = max(min(pred_bb[3], true_bb[3]) - max(pred_bb[1], true_bb[1]), 0)
    intersection_area = intersection_x*intersection_y
    union_area = pred_area + true_area - intersection_area

    if union_area > 0:
        return intersection_area/union_area
    else:
        return 0


def mota(ans_file, traj_true, traj_pred, threshold):
    corr = Correspondence(threshold = threshold)
    scores = {}
    for gt, pr in zip(traj_true, traj_pred):
        corr.count_fp_fn_idsw(gt, pr)
        for c, r in corr.temp_misses.items():
            if c not in scores:
                scores[c] = {'FP':0, 'FN':0, 'IDSW':0, 'GT':0}
            scores[c]['FP'] += corr.temp_misses[c]['FP']
            scores[c]['FN'] += corr.temp_misses[c]['FN']
            scores[c]['IDSW'] += corr.temp_misses[c]['IDSW']
            scores[c]['GT'] += corr.temp_misses[c]['GT']
    mota = 0
    gt_non_zero = 0
    for c,r in scores.items():
        if r['GT']>0:
            ss = 1 - (r['FP'] + r['FN'] + r['IDSW'])/r['GT']
            print(c, ss)
            print(c, "FP={}, FN={}, IDSW={}, GT={}".format(r["FP"], r["FN"], r["IDSW"], r["GT"]))

            # for global count
            if ans_file not in GLOBAL_SCORES or c not in GLOBAL_SCORES[ans_file]:
                GLOBAL_SCORES[ans_file] = {
                    'Pedestrian': {'FP': 0, 'FN': 0, 'IDSW': 0, 'GT': 0},
                    'Car': {'FP': 0, 'FN': 0, 'IDSW': 0, 'GT': 0}}
            GLOBAL_SCORES[ans_file][c]['FP'] += r["FP"]
            GLOBAL_SCORES[ans_file][c]['FN'] += r["FN"]
            GLOBAL_SCORES[ans_file][c]['IDSW'] += r["IDSW"]
            GLOBAL_SCORES[ans_file][c]['GT'] += r["GT"]
            mota += ss
            gt_non_zero += 1
        
    if gt_non_zero != 0:
        return mota/gt_non_zero
    else:
        return -10000


def MOTA(true_seqs, pred_seqs, threshold):
    ans_files = set(true_seqs).intersection(set(pred_seqs))
    s = 0
    for ans_file in ans_files:
        print(ans_file)
        traj_true = true_seqs[ans_file]
        traj_pred = pred_seqs[ans_file]
        s += mota(ans_file, traj_true, traj_pred, threshold)
        print('\n')

    return s/len(ans_files)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--prediction-file', default = 'predictions.json')
    parser.add_argument('--answer-file', default = 'ans.json')
    parser.add_argument('--threshold', default = 0.5)

    return parser.parse_args()


GLOBAL_SCORES = {}

def main():
    args = parse_args()

    with open(args.answer_file) as f:
        ans = json.load(f)

    with open(args.prediction_file) as f:
        sub = json.load(f)
    
    threshold = args.threshold
    
    score = MOTA(ans, sub, threshold)

    FP = 0; FN = 0; IDSW = 0; GT = 0
    for key in GLOBAL_SCORES.keys():
        if 'Pedestrian' in GLOBAL_SCORES[key].keys():
            print("{} {} {}".format(key, 'Ped',
                                    GLOBAL_SCORES[key]['Pedestrian']))
            FP += GLOBAL_SCORES[key]['Pedestrian']['FP']
            FN += GLOBAL_SCORES[key]['Pedestrian']['FN']
            IDSW += GLOBAL_SCORES[key]['Pedestrian']['IDSW']
            GT += GLOBAL_SCORES[key]['Pedestrian']['GT']
    print("------------------------")
    print("PED : FP={} FN={} IDSW={} GT={}".format(FP, FN, IDSW, GT))
    print("------------------------")

    FP = 0; FN = 0; IDSW = 0; GT = 0
    for key in GLOBAL_SCORES.keys():
        if 'Car' in GLOBAL_SCORES[key].keys():
            print("{} {} {}".format(key, 'Car',
                                    GLOBAL_SCORES[key]['Car']))
            FP += GLOBAL_SCORES[key]['Car']['FP']
            FN += GLOBAL_SCORES[key]['Car']['FN']
            IDSW += GLOBAL_SCORES[key]['Car']['IDSW']
            GT += GLOBAL_SCORES[key]['Car']['GT']
    print("------------------------")
    print("CAR : FP={} FN={} IDSW={} GT={}".format(FP, FN, IDSW, GT))
    print("------------------------")
    print(score)


if __name__ == '__main__':
    main()
