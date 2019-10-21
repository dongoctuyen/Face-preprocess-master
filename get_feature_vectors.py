import face_model
import argparse
import cv2
import sys
import os
import numpy as np
from vec2base64 import encode_array

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

dataset = '/home/comvision1/face-recognition/data/asian/asia_112x112'
with open('features.512.txt', 'a') as f:
    for id in os.listdir(dataset):
        count = 0
        images = os.listdir(os.path.join(dataset, id))
        if len(images) == 0:
            continue
        for image in images:
            img = cv2.imread(os.path.join(dataset, id, image))
            nimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            f1 = model.get_feature(aligned)
            f1 = f1[0]
            f.write('{"_id": "%s_%d", "_source":{"_aknn_vector":[%f' % (id, count, f1[0]))
            for feature in f1[1:]:
                f.write(',%f' % feature)
            f.write(']}}\n')
            count += 1

