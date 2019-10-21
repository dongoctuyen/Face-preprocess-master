from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import pickle as pkl
import sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from time import sleep
from easydict import EasyDict as edict
from time import gmtime, strftime
import time
sys.path.append(os.path.join(os.path.dirname(__file__)))
from mtcnn_detector import MtcnnDetector

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
import face_preprocess


class FaceModel:
    def __init__(self):
        self.threshold = 1.24
        self.det_minsize = 50
        self.det_threshold = [0.4, 0.6, 0.6]
        self.det_factor = 0.9
        image_size = '112,112'
        _vec = image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.image_size = image_size
        model = os.path.join(os.path.dirname(__file__), 'model-r100-ii/model,0')
        _vec = model.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        gpu = 0
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=mx.gpu(gpu), label_names=None)
        # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
        model.set_params(arg_params, aux_params)
        self.model = model
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.gpu(gpu), num_worker=3, accurate_landmark=True,
                                 threshold=[0.6, 0.7, 0.8])
        self.detector = detector
        self.det_option = 0

    def get_feature(self, face_img, detect=True):
        # face_img is BGR image
        if detect:
            ret = self.detector.detect_face(face_img)
            if ret is None:
                return None
            bboxs, points = ret
            if bboxs.shape[0] == 0:
                return None
            bbox = bboxs[0, 0:4]
            point = points[0, :].reshape((2, 5)).T
            nimg = face_preprocess.preprocess(face_img, bbox, point, image_size='112,112')
        else:
            nimg = face_img
            height, width, _ = face_img.shape
            if height != 112 or width != 112:
                nimg = cv2.resize(nimg, (112, 112))

        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))

        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        # print(_embedding.shape)
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding


if __name__ == '__main__':
    model = FaceModel()
