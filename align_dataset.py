import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
from face_preprocess import parse_lst_line, preprocess, preprocess_224


path = '/mnt/C89EC6E69EC6CBDE/computervision'
dataset = os.path.join(path, 'train_msra')
align_dataset = os.path.join(path, 'msra_112x96')
if not os.path.isdir(align_dataset):
    os.mkdir(align_dataset)

lmk_file = os.path.join(path, 'msra_lmk')
lst = open(os.path.join(path, 'msra_train_list.txt'), 'w')

with open(lmk_file, 'r') as f:
    while True:
        l = f.readline()
        vec = l.replace('\n', '').split(' ')
        img = vec[0].split('/')
        img_path = os.path.join(dataset, '%s/%s/%s'%(img[0], img[1], img[2]))
        img_align = os.path.join(align_dataset, '%s/%s'%(img[1], img[2]))

        temp = vec[-10:][0::2] + vec[-10:][1::2]

        point = np.array(temp)
        bbox = None
        point = point.reshape((2,5)).T
        if os.path.exists(img_path) and not os.path.exists(img_align):
            lst.write('%s/%s %s\n'%(img[1], img[2], vec[1]))
            image_arr = cv2.imread(img_path)
            if not os.path.isdir(os.path.join(align_dataset, '%s' % img[1])):
                os.mkdir(os.path.join(align_dataset, '%s' % img[1]))
            try:
                aligned = preprocess(image_arr, bbox, point, image_size='112,96')
                cv2.imwrite(img_align, aligned)
            except Exception as e:
                print(e)
                continue
