# -*- coding: utf-8 -*-  

import cv2
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes


class linemodcore(object):
    def __init__(self, so_file_path):
        """
        Init the dector get mem resource
        @param so_file_path: path to Dynamic Link Library 
        """
        self.dll = ctypes.CDLL(so_file_path)
        self.dll.img_quant.restype = ctypes.POINTER(ctypes.c_uint8)
        self.dll.match_target.restype = ctypes.POINTER(ctypes.c_char_p)
        # 新版core
        self.dll.initDector()

    def cal_quant(self, img):
        """
        Get a imgs Gradient direction encode
        @param img: bgr img as numpy types
        """
        i_b, i_g, i_r = cv2.split(img)
        sH, sW = i_b.shape[0], i_b.shape[1]
        frame_b = np.asarray(i_b, dtype=np.uint8)
        frame_b = frame_b.ctypes.data_as(ctypes.c_char_p)
        frame_g = np.asarray(i_g, dtype=np.uint8)
        frame_g = frame_g.ctypes.data_as(ctypes.c_char_p)
        frame_r = np.asarray(i_r, dtype=np.uint8)
        frame_r = frame_r.ctypes.data_as(ctypes.c_char_p)
        ag_pointer = self.dll.img_quant(sH, sW, frame_b, frame_g, frame_r)
        np_ag = np.array(np.fromiter(ag_pointer, dtype=np.uint8, count=sH*sW))
        return np_ag.reshape((sH,sW))

    def load_a_temp(self, single_tmp_get):
        """
        Add a template to Dector
        @param single_tmp_get: a string of single template
        """
        self.dll.Dector_loadTep(ctypes.c_char_p(bytes(single_tmp_get, 'ascii')))

    def load_a_id(self, single_id_get):
        """
        Add a id that need to dect
        @param single_id_get: a string of a single id
        """
        self.dll.Dector_setId(ctypes.c_char_p(bytes(single_id_get, 'ascii')))

    def clean_id(self):
        """
        Clean all id
        """
        self.dll.Dector_cleanId()

    def match_a_img(self, img_get):
        """
        Match a single image
        @param img_get: target image as numpy types
        @return re_list: a list of resault, empty if no match
        """
        i_b, i_g, i_r = cv2.split(img_get)
        sH, sW = i_b.shape[0], i_b.shape[1]
        frame_b = np.asarray(i_b, dtype=np.uint8)
        frame_b = frame_b.ctypes.data_as(ctypes.c_char_p)
        frame_g = np.asarray(i_g, dtype=np.uint8)
        frame_g = frame_g.ctypes.data_as(ctypes.c_char_p)
        frame_r = np.asarray(i_r, dtype=np.uint8)
        frame_r = frame_r.ctypes.data_as(ctypes.c_char_p)
        ag_pointer = self.dll.match_target(sH, sW, frame_b, frame_g, frame_r)
        buf_string = ctypes.string_at(ag_pointer, -1).decode("ascii")
        re_list = []
        if buf_string != "NULL":
            for i in buf_string.splitlines():
                single_match = i.split(' ')
                re_list.append([single_match[0], float(single_match[1]), int(single_match[2]), int(single_match[3]), int(single_match[4])])
        return re_list

### 以下为 新版core 需要的功能
    def add_single_shape(self, id_get, area_get):
        total_list = []
        for i in range(360):
            tmp_list = []
            for j in area_get[i]:
                tmp_list.append(j[0])
                tmp_list.append(j[1])
            tmp_arr = (ctypes.c_int16*len(tmp_list))(*tmp_list)
            total_list.append(tmp_arr)
        area_arr = ((ctypes.c_int16*(len(area_get[0])*2))*360)(*total_list)
        self.dll.add_a_shape(ctypes.c_char_p(bytes(id_get, 'ascii')), area_arr, ctypes.c_int16(len(area_get[0])*2))

    def print_shape(self, id_print, angle_print):
        self.dll.print_a_shape(ctypes.c_char_p(bytes(id_print, 'ascii')), ctypes.c_int16(angle_print))

    def clean_shape(self):
        self.dll.clean_shape()
    def set_nms(self):
        self.dll.SetNms()