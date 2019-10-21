import cv2
import mxnet as mx
import numpy as np
import os
import argparse
import sys
import shutil
from scipy.spatial.distance import cosine
# sys.path.append(os.path.join(os.path.dirname(__file__)))
from mtcnn_detector import MtcnnDetector
from face_embedding import FaceModel
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess


def normal_filter(input_dir, output_dir):
    gpu = 0
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.gpu(gpu), num_worker=3, accurate_landmark=True,
                             threshold=[0.6, 0.7, 0.8])
    threshold = 0.95

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for image in os.listdir(input_dir):
        # print(image)
        try:
            img_path = os.path.join(input_dir, image)
            face_img = cv2.imread(img_path)
            ret = detector.detect_face(face_img)
            if ret is None:
                continue
            bboxs, points = ret
            # print bboxs, points
            if len(bboxs) > 1:
                centers = []
                height, width, _ = face_img.shape
                center = [int(height / 2), int(width / 2)]
                for bbox in bboxs:
                    if bbox[4] < threshold:
                        continue
                    centers.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                dist2center = np.sum(np.square(np.array(centers) - center), axis=1)
                closest_index = np.argmin(dist2center)
                bbox = bboxs[closest_index, 0:4]
                point = points[closest_index, :].reshape((2, 5)).T
            elif bboxs.shape[0] == 0:
                continue
            else:
                bbox = bboxs[0, 0:4]
                point = points[0, :].reshape((2, 5)).T
            nimg = face_preprocess.preprocess(face_img, bbox, point, image_size='112,112')
            cv2.imwrite(os.path.join(output_dir, image), nimg)
        except Exception as e:
            print(e)

def logic_filter(input_dir, output_dir):
    face_model = FaceModel()
    ids = os.listdir(input_dir)
    threshold = 0.5

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for id in ids:
        images = os.listdir(os.path.join(input_dir, id))
        if len(images)<5:
            continue
        embeddings = []
        for image in images:
            face_img = cv2.imread(os.path.join(input_dir, id, image))
            res = face_model.get_feature(face_img, detect=False)
            embeddings.append(res)

        pattern = None
        for i, embedding in enumerate(embeddings):
            count = 0
            for j in range(len(embeddings)):
                if j != i and 1-cosine(embedding, embeddings[j])>threshold:
                    count += 1
            if count/len(embeddings)>=0.5:
                pattern = embedding
                break
        if pattern is not None:
            for i, embedding in enumerate(embeddings):
                if 1-cosine(embedding, pattern)<threshold:
                    if not os.path.isdir(os.path.join(output_dir, id)):
                        os.mkdir(os.path.join(output_dir, id))
                    shutil.move(os.path.join(input_dir, id, images[i]), os.path.join(output_dir, id, images[i]))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--input_dir', default='./face_ids', help='Input folder faces image')
    parser.add_argument('--output_dir', default='./filted_images', help='Output folder')
    parser.add_argument('--logic_filter', type=bool, default=False, help='Output folder')
    args = parser.parse_args()

    is_logic = args.logic_filter
    if not is_logic:
        normal_filter(args.input_dir, args.output_dir)
    else:
        logic_filter(args.input_dir, args.output_dir)
