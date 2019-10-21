import shutil
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--input_dir', default='./images', help='Input folder faces image')
    parser.add_argument('--output_dir', default='./face_ids', help='Output folder')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for image in os.listdir(args.input_dir):
        img_path = os.path.join(args.input_dir, image)
        info = image.split('_')
        fb_id = info[0]
        if not os.path.isdir(os.path.join(args.output_dir, fb_id)):
            os.mkdir(os.path.join(args.output_dir, fb_id))
        shutil.copyfile(img_path, os.path.join(args.output_dir, fb_id, '%s_%s'%(info[1], info[2])))