# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import copy
from shapely.geometry import Polygon


HOG_MAG_thr = 25
HOG_SIM_thr = 0.75
HIST_SIM_thr = 0.95

def iou_mask(poly_1, poly_2):
    if poly_1.intersects(poly_2):
        return poly_1.intersection(poly_2).area
    else:
        return 0

def area_color_process(tf_get, ang_img, img_get, res_get, bi_img = None):
    re_res = []
    back_list = []
    back_img = np.zeros(img_get.shape[:-1], np.uint8)
    img_hsv = cv2.cvtColor(img_get, cv2.COLOR_BGR2HSV)
    type_list = list(set([k[0] for k in res_get]))
    flag_list = [True]*len(res_get)
    area_dic = {}
    for i in type_list:
        area_dic[i] = Polygon(tf_get[i].area[0]).area
    sort_area_list = sorted(area_dic.keys(), key = lambda item: area_dic[item],reverse=True)
    for _, i in enumerate(sort_area_list):
        for j_idx,j in enumerate(res_get):
            if j[0] == i and flag_list[j_idx]:
                flag_list[j_idx] = False
                cen_x, cen_y = j[2], j[3]
                curr_polygon = np.array([[[k[0]+cen_x, k[1]+cen_y] for k in tf_get[i].area[j[4]]]])
                curr_shape = Polygon(curr_polygon[0])
                pass_flg = True
                for k in back_list:
                    if iou_mask(curr_shape, k)*4 >= area_dic[i]:
                        pass_flg = False
                        break
                if pass_flg:
                    curr_draw = copy.deepcopy(back_img)
                    cv2.fillPoly(curr_draw, curr_polygon, 255)
                    x, y, w, h = cv2.boundingRect(curr_polygon)
                    if x>=0:
                        x1 = x
                    else:
                        x1=0
                    if (x+w) <= back_img.shape[1]:
                        x2 = x+w
                    else:
                        x2 = back_img.shape[1]
                    if y>= 0:
                        y1 = y
                    else:
                        y1 = 0
                    if (y+h) <= back_img.shape[0]:
                        y2 = y+h
                    else:
                        y2 = back_img.shape[0]
                    if hist_test(img_hsv, tf_get[i].color, curr_draw):
                        back_list.append(curr_shape)
                        re_res.append(j)
                    del curr_draw
    return re_res



def hist_test(img_get, hist_get, mask_get):
    global HIST_SIM_thr
    co_list = []
    tmp_hist = cv2.calcHist([img_get], [0], mask_get, [10], [0,180])
    for j in range(10):
        co_list.append(int(tmp_hist[j]))
    for k in range(1,3):
        tmp_hist = cv2.calcHist([img_get], [k], mask_get, [10], [0,255])
        for j in range(10):
            co_list.append(int(tmp_hist[j]))
    img_hist = np.array(co_list)
    tmp_hist = np.array(hist_get)
    sim_res_1 = cos_sim(img_hist[0:10], tmp_hist[0:10])
    sim_res_2 = cos_sim(img_hist[10:20], tmp_hist[10:20])
    sim_res_3 = cos_sim(img_hist[20:30], tmp_hist[20:30])
    thr_a = 0.95
    if (sim_res_1>HIST_SIM_thr) and (sim_res_2>HIST_SIM_thr) and (sim_res_3>HIST_SIM_thr):
        return True
    else:
        return False

def cos_sim(v_a, v_b):
    return np.dot(v_a, v_b)/(np.linalg.norm(v_a)*np.linalg.norm(v_b))

def match_single_img(tf, raw_res, img_ag, img, bi_img = None):
    if not raw_res == []:
        nms_res = nms_process(raw_res, 15)
        if bi_img is None:
            area_res = area_color_hog_process(tf, img_ag, img, nms_res)
        else:
            area_res = area_color_hog_process(tf, img_ag, img, nms_res, bi_img)
        if not area_res == []:
            re_res = catch_process(tf, area_res)
            return [re_res, area_res]
        else:
            return []
    return []


    



