import cv2
import os
import sys
import mxnet as mx
from mtcnn_detector import MtcnnDetector

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess

gpu = 0
mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.gpu(gpu), num_worker=3, accurate_landmark=True,
                         threshold=[0.6, 0.7, 0.8])
path = '/mnt/C89EC6E69EC6CBDE/computervision/'
root = os.path.join(path, 'lfw-deepfunneled')
dst_path = os.path.join(path, 'lfw-112x96')
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
ids = os.listdir(root)
for id in ids:
    images = os.listdir(os.path.join(root, id))
    # print images
    if not os.path.exists(os.path.join(dst_path, id)):
        os.mkdir(os.path.join(dst_path, id))
    for image in images:
        if os.path.exists(os.path.join(dst_path, id, image)):
            continue
        if image.split('.')[-1] not in ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']:
            continue
        img_arr = cv2.imread(os.path.join(root, id, image))
        ret = detector.detect_face(img_arr)
        if ret is None:
            continue
        bboxs, points = ret
        bbox = bboxs[0, 0:4]
        point = points[0, :].reshape((2, 5)).T
        aligned = face_preprocess.preprocess(img_arr, bbox, point, image_size='112,96')
        cv2.imwrite(os.path.join(dst_path, id, image), aligned)
