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
import PIL.Image
import os

label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

classTitle = ['person_bmp', 'person_poly', 'neutral']


class SuperviselyPersonDatasetPrepare:

    def __init__(self, parent_path: str, output_path: str):
        self._parrent_path = parent_path
        self._output_path = output_path

        self._output_path_train = os.path.join(output_path, "train")
        self._output_path_val = os.path.join(output_path, "val")
        self._output_path_ann_train = os.path.join(output_path, "train\\ann")
        self._output_path_img_train = os.path.join(output_path, "train\\img")
        self._output_path_ann_val = os.path.join(output_path, "val\\ann")
        self._output_path_img_val = os.path.join(output_path, "val\\img")

        if os.path.exists(self._output_path) is False:
            os.mkdir(self._output_path)

        if os.path.exists(self._output_path_train) is False:
            os.mkdir(self._output_path_train)

        if os.path.exists(self._output_path_val) is False:
            os.mkdir(self._output_path_val)

        if os.path.exists(self._output_path_ann_train) is False:
            os.mkdir(self._output_path_ann_train)

        if os.path.exists(self._output_path_img_train) is False:
            os.mkdir(self._output_path_img_train)

        if os.path.exists(self._output_path_ann_val) is False:
            os.mkdir(self._output_path_ann_val)

        if os.path.exists(self._output_path_img_val) is False:
            os.mkdir(self._output_path_img_val)

        temp = ()
        for it in label_colours:
            temp += it

        self._png_palette = list(temp)
        self.count = 0

    def base64_2_mask(self,s):
        z = zlib.decompress(base64.b64decode(s))
        n = np.fromstring(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3]  # rgba 4 chanels
        return mask

    def generate_mask_bmp(self, bitmap: dict, image_zero, color_index):

        origin = bitmap["origin"]  # x and y coordiante
        bit_data = bitmap["data"]
        imgdata = self.base64_2_mask(bit_data)
        for i in range(imgdata.shape[0]):  # origin[1]:origin[1]+
            for j in range(imgdata.shape[1]):
                if imgdata[i, j] > 0:
                    image_zero[origin[1] + i, origin[0] + j] = color_index

    def list_to_image_data(self, image_out, list_data: list, color_index):

        if isinstance(list_data[0][0], list):
            # multiple list data
            for ele in list_data:
                exterior = np.array(ele).astype(np.int32)
                cv2.fillPoly(image_out, [exterior], (color_index))
        else:
            exterior = np.array(list_data).astype(np.int32)
            cv2.fillPoly(image_out, [exterior], (color_index))


    def generate_mask_poly(self, points: dict, image_zero, color_index):

        width = image_zero.shape[1]
        height = image_zero.shape[0]
        blank_image_exterior = image_zero #np.zeros((height, width, 1), np.uint8)
        blank_image_interior = np.zeros((height, width, 1), np.uint8)
        exterior = points["exterior"]
        interior = points["interior"]
        if len(exterior) > 0:
            self.list_to_image_data(blank_image_exterior, exterior, color_index)
        if len(interior) > 0:
            self.list_to_image_data(blank_image_interior, interior, color_index)
        blank_image_interior = color_index - blank_image_interior
        cv2.bitwise_and(blank_image_interior, blank_image_exterior, image_zero)


    def internal_process(self, img_path, ann_path):
        print(img_path)
        with open(ann_path, 'r') as f:
            data = json.load(f)

        img_png = PIL.Image.open(img_path)
        image_zero = np.zeros((img_png.height, img_png.width), dtype=np.uint8)

        for obj in data["objects"]:
            if (obj["classTitle"] == classTitle[0]):
                self.generate_mask_bmp(obj["bitmap"], image_zero, 1)
            elif(obj["classTitle"] == classTitle[1]):
                self.generate_mask_poly(obj["points"], image_zero, 1)
            else:
                print("here is {0} ".format(obj["classTitle"]))

        # cv2.imshow("image_zero", image_zero)
        # cv2.waitKey(0)
        # save png in index mode which is "p" mode
        png = Image.fromarray(image_zero)
        png.putpalette(self._png_palette)
        base_output_name = os.path.basename(img_path)
        self.count += 1
        if self.count % 8 == 0:
            png.save(os.path.join(self._output_path_ann_val, base_output_name))
            img_png.save(os.path.join(self._output_path_img_val, base_output_name))
        else:
            png.save(os.path.join(self._output_path_ann_train, base_output_name))
            img_png.save(os.path.join(self._output_path_img_train, base_output_name))


    def generate_index_png(self):

        for idx in range(1, 14, 1):
            sub_folder_ann = os.path.join(self._parrent_path, "ds{0}\\ann".format(idx))
            sub_folder_img = os.path.join(self._parrent_path, "ds{0}\\img".format(idx))
            for r, d, f in os.walk(sub_folder_img):
                for file in f:
                    if ".png" in file:
                        img_file_path = os.path.join(r, file)
                        # check json file
                        ann_file_name = os.path.basename(file) + ".json"
                        ann_file_path = os.path.join(sub_folder_ann, ann_file_name)

                        if os.path.exists(ann_file_path):
                            self.internal_process(img_file_path, ann_file_path)


if __name__ == "__main__":

    # pexels-photo-358010.png pexels-photo-237705
    dataset = SuperviselyPersonDatasetPrepare("G:\\imageDatasets\\Supervisely Person Dataset\\",
                                              "G:\\imageDatasets\\Supervisely Person Dataset\\outputs\\")

    dataset.generate_index_png()

    # dataset.internal_process("G:\\imageDatasets\\Supervisely Person Dataset\\ds9\\img\\pexels-photo-864994.png",
    #                          "G:\\imageDatasets\\Supervisely Person Dataset\\ds9\\ann\\pexels-photo-864994.png.json")

    print("dataset done")





