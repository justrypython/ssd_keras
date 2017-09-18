#encoding:UTF-8

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras_ssd7 import build_model
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator
import os
import cv2

### Set up the model

# 1: Set some necessary parameters

img_height = 300 # Height of the input images
img_width = 480 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 2 # Number of classes including the background class
min_scale = 0.08 # The scaling factor for the smallest anchor boxes
max_scale = 0.96 # The scaling factor for the largest anchor boxes
scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = False # Whether or not the model is supposed to use relative coordinates that are within [0,1]

# 2: Build the Keras model (and possibly load some trained weights)

K.clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
model, predictor_sizes = build_model(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                      min_scale=min_scale,
                                      max_scale=max_scale,
                                      scales=scales,
                                      aspect_ratios_global=aspect_ratios,
                                      aspect_ratios_per_layer=None,
                                      two_boxes_for_ar1=two_boxes_for_ar1,
                                      limit_boxes=limit_boxes,
                                      variances=variances,
                                      coords=coords,
                                      normalize_coords=normalize_coords)
#model.load_weights('./ssd7_0_weights.h5')
#model = load_model('./ssd7_0.h5')

model.load_weights('./model_v2_5.h5')

real_img_path = r'E:\justrypython\ks_idcard_ocr\testimg/card_bat/'

paths = []
for dirpath, dirnames, filenames in os.walk(real_img_path):
    for filename in filenames:
        filepath = dirpath + os.sep +filename
        paths.append(filepath)


def plot(path):
    X = cv2.imread(path)
    newimg = cv2.resize(X, (300, 300), interpolation=cv2.INTER_CUBIC)
    bg_img = np.zeros((300, 480, 3), dtype=np.uint8)
    bg_img[:, 90:390] = newimg
    for k in range(90):
        bg_img[:, k] = newimg[:, 0]
    for k in range(90):
        bg_img[:, 390+k] = newimg[:, -1]
    bg_img = bg_img.reshape((1, )+bg_img.shape)
    # print bg_img.shape
    y_pred = model.predict(bg_img)
    # print y_pred
    y_pred_decoded = decode_y2(y_pred,
                           confidence_thresh=0.1,
                          iou_threshold=0.1,
                          top_k='all',
                          input_coords='centroids',
                          normalize_coords=False,
                          img_height=None,
                          img_width=None)
    #print y_pred_decoded
    plt.figure(figsize=(20,12))
    plt.imshow(bg_img[0])

    current_axis = plt.gca()

    #classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
    classes = ['background', 'card'] # Just so we can print class names onto the image instead of IDs

    # Draw the predicted boxes in blue
    for box in y_pred_decoded[0]:
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
        current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})
    plt.show()

for k, j in enumerate(paths):
    print (j)
    plot(j)