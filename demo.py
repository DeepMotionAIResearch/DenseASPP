#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: {maokeyang, kunyu, kuiyuanyang}@deepmotion.ai
# Inference code of DenseASPP based segmentation models.


import argparse
from inference import Inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DenseASPP inference code.')
    parser.add_argument('--model_name', default='DenseASPP161', help='segmentation model.')
    parser.add_argument('--model_path', default='./weights/denseASPP161.pkl', help='weight path.')
    parser.add_argument('--img_dir', default='./Cityscapes/leftImg8bit/val', help='image dir.')
    args = parser.parse_args()

    infer = Inference(args.model_name, args.model_path)
    infer.folder_inference(args.img_dir, is_multiscale=False)
