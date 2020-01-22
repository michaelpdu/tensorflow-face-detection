#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

import os
import argparse
import numpy as np
import skimage.io as io
from skimage.transform import rescale

def resize_image(img, desired_size, expand, uint8_range):
    """
    Resize image.

    Parameters:
        img - image data, which is (H,W,C)
        desired_size - (H,W), None indicates original size
        expand - True is to expand dimention, False will not
        uint8_range - True indicates [0,255], False indicates [0.,1.]
    Returns:
        return a float numpy ndarray
    Notes:
        None
    """
    if desired_size:
        # Separate scale factors can be defined as (row_scale, col_scale)
        factors = (desired_size[0]/float(img.shape[0]), desired_size[1]/float(img.shape[1]))
    else:
        factors = (1.0, 1.0)
    # print('factors:', factors)
    img = rescale(img, factors, preserve_range=uint8_range, order=5)
    if uint8_range:
        img = img.astype(np.uint8)
    # print('after rescale, img:', img)
    # print('after rescale, shape:', img.shape)
    if expand:
        img = np.expand_dims(img, axis=0)
    return img

def img2array(data_path, desired_size=None, expand=False, uint8_range=False, view=False):
    """
    Loads an RGB image as a 3D or 4D numpy array.

    Parameters:
        data_path - image file path
        desired_size - (H,W), None indicates original size
        expand - True is to expand dimention, False will not
        uint8_range - True indicates [0,255], False indicates [0.,1.]
        view - show image or not
    Returns:
        return a float numpy ndarray
    Notes:
        None
    """
    img = io.imread(data_path) # return (H,W,C)
    # print('original image:', img)
    img = resize_image(img, desired_size, expand, uint8_range)
    if view:
        io.imshow(img)
    return img

def inference_image(img_path):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            # # 
            # ops = tf.get_default_graph().get_operations()
            # for op in ops:
            #     print(op.name) # print operation name
            #     print('> ', op.values())

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=detection_graph, config=config) as sess:
            #
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            start_time = time.time()
            imgs = img2array(img_path, desired_size=None, expand=True, uint8_range=False)
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: imgs})
            elapsed_time = time.time() - start_time
            print('inference time cost: {}'.format(elapsed_time))
            print(boxes.shape, boxes)
            print(scores.shape,scores)
            print(classes.shape,classes)
            print(num_detections)

_examples = """
  
  # predict face probability in image
  python %(prog)s ./images/base/aligned/xinchi_aligned.png

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('img_path', type=str, help='Path to the image.')
    args = parser.parse_args()

    inference_image(args.img_path)