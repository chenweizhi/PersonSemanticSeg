#!/usr/bin/env python
# -*- coding: utf-8 -*-


from pandas.io.json import json_normalize
import pandas as pd
import json
import time
import cv2
import base64
from PIL import Image
import io
import numpy as np
import zlib

classTitle = ['person_bmp', 'person_poly']

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3] #rgba 4 chanels
    return mask

def generate_mask_bmp(bitmap: dict):
    origin = bitmap["origin"]
    bit_data = bitmap["data"]
    imgdata = base64_2_mask(bit_data)

    cv2.imshow("imgdata", imgdata)
    cv2.waitKey(0)

def generate_mask_poly(points: dict, size: dict):
    width = size["width"]
    height = size["height"]
    blank_image_exterior = np.zeros((height, width, 1), np.uint8)
    blank_image_interior = np.zeros((height, width, 1), np.uint8)
    exterior = np.array(points["exterior"]).astype(np.int32)
    interior = np.array(points["interior"]).astype(np.int32)
    cv2.fillPoly(blank_image_exterior, [exterior], (255))
    cv2.fillPoly(blank_image_interior, [interior], (255))
    blank_image_interior = 255 - blank_image_interior
    cv2.bitwise_and(blank_image_interior, blank_image_exterior, blank_image_exterior)
    cv2.imshow("blank_image", blank_image_exterior)
    cv2.waitKey(0)

if __name__ == "__main__":

    image = "G:\\imageDatasets\\Supervisely Person Dataset\\ds13\\ann\\pexels-photo-358010.png.json"
    image2 = "G:\\imageDatasets\\Supervisely Person Dataset\\ds13\\ann\\pexels-photo-237705.png.json"
    with open(image, 'r') as f:
        data = json.load(f)

    with open(image2, 'r') as f:
        data2 = json.load(f)

    tmp_data = data2
    if(tmp_data["objects"][0]["classTitle"] == classTitle[0]):
        generate_mask_bmp(tmp_data["objects"][0]["bitmap"])
    else:
        generate_mask_poly(tmp_data["objects"][0]["points"], tmp_data["size"])


