import json
import os
import pdb

train_annotations_path = os.path.join("./train_annotations")
train_annotations_files = os.listdir(train_annotations_path)
ans_converted = {}


def check_dict(x):
    res = {}
    res2 = {}
    # pdb.set_trace()
    try:
        if "Pedestrian" in x.keys():
            res['Pedestrian'] = x['Pedestrian']
            res2['Pedestrian'] = []
            for ped in res['Pedestrian']:
                ped_box = ped['box2d']
                area = (ped_box[2] - ped_box[0]) * (ped_box[3] - ped_box[1])
                if area <= 1024:
                    continue
                else:
                    res2['Pedestrian'].append(ped)

        if "Car" in x.keys():
            res['Car'] = x['Car']
            res2['Car'] = []
            for ped in res['Car']:
                ped_box = ped['box2d']
                area = (ped_box[2] - ped_box[0]) * (ped_box[3] - ped_box[1])
                if area <= 1024:
                    continue
                else:
                    res2['Car'].append(ped)
    except:
        pdb.set_trace()
        print("hey")
    return res2


def convert_to_ans_json(train_annotation_file):
    file_name = train_annotation_file.replace('./train_annotations/', '').replace('.json', '')
    with open(train_annotation_file) as f:
        train_json = json.load(f)
        filtered_list = [check_dict(x) for x in train_json['sequence']]
        file_name = file_name + ".mp4"
        ans_converted[file_name] = filtered_list


for train_annotation_file in train_annotations_files:
    if train_annotation_file.split('.')[-1] != 'json':
        continue

    print("Converting file {}".format(train_annotation_file))
    convert_to_ans_json(train_annotations_path + "/" + train_annotation_file)


ans_file = 'ans_converted.json'
with open(ans_file, 'w') as outfile:
    print("Dumping output to {}".format(ans_file))
    json.dump(ans_converted, outfile)