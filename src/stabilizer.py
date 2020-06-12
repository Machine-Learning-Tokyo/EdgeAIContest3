import cv2
import os
import numpy as np
from glob import glob
from argparse import ArgumentParser



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, dest="input_dir", required=True, help="input directory path")
    parser.add_argument("-o", "--output", type=str, dest="output_dir", required=True, help="output directory path")
    args = parser.parse_args()

    images = glob(os.path.join(args.input_dir, '*'))
    images.sort()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    first_image = None
    prev_flow = None
    total_cost = 0
    total_mag = 0
    for n, image in enumerate(images):
        print(n)
        rgb_image = cv2.imread(image)
        rgb_image = cv2.resize(rgb_image, (rgb_image.shape[1]//4, rgb_image.shape[0]//4), interpolation=cv2.INTER_CUBIC)
        oh, ow = rgb_image.shape[:2]
        rgb_image = rgb_image[:oh*4//5, :, :]
        oh, ow = rgb_image.shape[:2]
        pad_w = 16
        th, tw = (oh+pad_w//2-1)//(pad_w//2)*(pad_w//2), (ow+pad_w//2-1)//(pad_w//2)*(pad_w//2)
        rgb_image = cv2.resize(rgb_image, (tw, th), interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        if first_image is None:
            first_image = gray_image
            px1, py1, px2, py2 = 0, 0, tw, th
            continue
        second_image = gray_image

        lx, ly, ux, uy = -pad_w, 0, tw+pad_w, th
        cx1, cy1, cx2, cy2 = px1, py1, px2, py2
        first_pad_image = np.zeros((th, tw+2*pad_w))
        first_pad_image[:, pad_w:pad_w+tw] = first_image
        mask = np.zeros((th, tw+2*pad_w))
        mask[:, pad_w:pad_w+tw] = 1
        xx, yy = np.meshgrid(range(tw+2*pad_w), range(th))
        ttw = tw + 2*pad_w
        crop1 = np.logical_and(ttw*yy<6*th*(xx-ttw//4), ttw*yy<-6*th*(xx-ttw+ttw//4))
        crop2 = np.logical_and(ttw*(yy-th//2)<th*(xx-ttw//3), ttw*(yy-th//2)<-th*(xx-ttw+ttw//3))
        crop = np.logical_and(crop1, crop2)
        # first_pad_image[crop] = 0
        mask[crop] = 0

        def get_cost(x1, y1, x2, y2):
            image1 = first_pad_image[y1:y2, x1+pad_w:x2+pad_w]
            mask1 = mask[y1:y2, x1+pad_w:x2+pad_w]
            image2 = cv2.resize(second_image, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_CUBIC)
            cost = np.sum(np.abs(image1-image2)*mask1)
            nonzero = np.count_nonzero(mask1)
            zero = mask1.shape[0]*mask1.shape[1] - nonzero
            cost = cost/nonzero
            return cost, image1, image2

        min_cost, _, _ = get_cost(cx1, cy1, cx2, cy2)
        visit = set([(cx1, cy1, cx2, cy2)])

        for step in [8, 4, 4, 2, 2, 1, 1, 1]:
            bx1, by1, bx2, by2 = cx1, cy1, cx2, cy2
            for dy1 in [-step, 0, step]:
                for dy2 in [-step, 0, step]:
                    ny1 = cy1 + dy1
                    ny2 = cy2 + dy2
                    if ny1<ly or ny2>uy: continue
                    for dx1 in [-step, 0, step]:
                        for dx2 in [-step, 0, step]:
                            if (dy1, dy2, dx1, dx2) == (0, 0, 0, 0): continue
                            nx1 = cx1 + dx1
                            nx2 = cx2 + dx2
                            if (nx1, ny1, nx2, ny2) in visit: continue
                            visit.add((nx1, ny1, nx2, ny2))
                            if nx1<lx or nx2>ux: continue
                            cost, _, _ = get_cost(nx1, ny1, nx2, ny2)
                            if cost<min_cost:
                                min_cost = cost
                                bx1, by1, bx2, by2 = nx1, ny1, nx2, ny2
            cx1, cy1, cx2, cy2 = bx1, by1, bx2, by2

        total_cost += min_cost
        total_mag += abs(cx1) + abs(cy1) + abs(cx2-tw) + abs(cy2-th)
        print(cx1, cy1, cx2, cy2, min_cost, total_cost/n, total_mag/n)
        px1, py1, px2, py2 = cx1, cy1, cx2, cy2
        _, image1, image2 = get_cost(cx1, cy1, cx2, cy2)
        cv2.imwrite(os.path.join(args.output_dir, f'{n:05d}_1.png'), first_image)
        cv2.imwrite(os.path.join(args.output_dir, f'{n:05d}_2.png'), cv2.resize(image1, (tw, th), interpolation=cv2.INTER_CUBIC))
        cv2.imwrite(os.path.join(args.output_dir, f'{n:05d}_3.png'), second_image)

        first_image = second_image
