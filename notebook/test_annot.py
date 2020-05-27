from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

def main():
    # Load annotation
    path_coco_json = "../data/COCO_annotations/train_09.json"
    path_to_image = '../data/train_images/'
    example_coco = COCO(path_coco_json)

    categories = example_coco.loadCats(example_coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

    category_names = set([category['supercategory'] for category in categories])
    print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

    # load annotation
    print(example_coco.getCatIds(catNms=['Car']))
    print(example_coco.getImgIds(catIds = [1,2]))

    


if __name__ == "__main__":
    main()