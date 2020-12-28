# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import copy



def eva_hog_fea(temp_hog_info, cen_x, cen_y, img_norm, img_ang):
    total_patch_num = 0
    re_res = []
    for i in temp_hog_info:
        tmp_pos_y = cen_y+i[1]
        tmp_pos_x = cen_x+i[0]
        scene_hog = np.zeros((8,), np.float16)
        for j in range(8):
            scene_hog[j] = np.sum(img_norm[tmp_pos_y-7:tmp_pos_y+8, tmp_pos_x-7:tmp_pos_x+8][np.where(img_ang[tmp_pos_y-7:tmp_pos_y+8, tmp_pos_x-7:tmp_pos_x+8] == j)])
        sim = cos_sim(scene_hog, np.array(i[2]))
        if not(sim >= 0 or sim <= 1):
            import pdb
            pdb.set_trace()

        sum_tmp_t = np.sum(i[2])
        sum_tmp_s = np.sum(scene_hog)
        re_res.append(abs(sum_tmp_t - sum_tmp_s))
        re_res.append(sim)
    return re_res

def get_img_norm(img_input):
    blur_img = cv2.GaussianBlur(img_input, (3,3), 0,0)
    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1,0, ksize = 3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0,1, ksize = 3)
    img_mag = cv2.magnitude(sobel_x, sobel_y)
    img_mag[img_mag >255] = 255
    img_norm = img_mag/255
    return img_norm

def fill_res(tf_get, ang_img, img_get, res_get, bi_img = None):
    fac = len(res_get)
    re_res = []
    back_img = np.zeros(img_get.shape[:-1], np.uint8)
    color_back_img = np.zeros(img_get.shape, np.uint8)
    draw_img = np.zeros(img_get.shape[:-1], np.uint8)
    img_hsv = cv2.cvtColor(img_get, cv2.COLOR_BGR2HSV)
    norm_img = get_img_norm(img_get)
    for nk,j in enumerate(res_get):
        print("\r#>", str(nk)+'/'+str(fac), str(nk/fac), end='')
        cen_x, cen_y = j[2], j[3]
        curr_polygon = np.array([[[k[0]+cen_x, k[1]+cen_y] for k in tf_get[j[0]].area[j[4]]]])
        curr_draw = copy.deepcopy(draw_img)
        cv2.fillPoly(curr_draw, curr_polygon, 255)
        hist_info = hist_test(img_hsv, tf_get[j[0]].color, curr_draw)
        hog_info = eva_hog_fea(tf_get[j[0]].obvious[j[4]], cen_x, cen_y, norm_img, ang_img)
        re_res.append(j+hist_info+hog_info)
    print('')
    return re_res

def f_res_mp(res_head):
    j = res_head[0]
    tf_get = res_head[1]
    draw_img = res_head[2]
    img_hsv = res_head[3]
    norm_img = res_head[4]
    ang_img = res_head[5]
    cen_x, cen_y = j[2], j[3]
    curr_polygon = np.array([[[k[0]+cen_x, k[1]+cen_y] for k in tf_get[j[0]].area[j[4]]]])
    curr_draw = copy.deepcopy(draw_img)
    cv2.fillPoly(curr_draw, curr_polygon, 255)
    hist_info = hist_test(img_hsv, tf_get[j[0]].color, curr_draw, res_head[6])
    hog_info = eva_hog_fea(tf_get[j[0]].obvious[j[4]], cen_x, cen_y, norm_img, ang_img)
    return j+hist_info+hog_info

def fill_res_mp(tf_get, ang_img, img_get, res_get, mode = 'HSV'):
    print("#>",len(res_get))
    back_img = np.zeros(img_get.shape[:-1], np.uint8)
    color_back_img = np.zeros(img_get.shape, np.uint8)
    draw_img = np.zeros(img_get.shape[:-1], np.uint8)
    if mode == 'HSV':
        img_hsv = cv2.cvtColor(img_get, cv2.COLOR_BGR2HSV)
        hsv_flg = True
    else:
        img_hsv = copy.deepcopy(img_get)
        hsv_flg = False
    norm_img = get_img_norm(img_get)
    itmes = [[i,tf_get,draw_img,img_hsv,norm_img,ang_img, hsv_flg] for i in res_get]
    import multiprocessing
    p = multiprocessing.Pool(6)
    b = p.map(f_res_mp, itmes)
    p.close()
    p.join()
    return list(b)




def hist_test(img_get, hist_get, mask_get, hsv_flg=True):
    co_list = []
    #debug
    if hsv_flg:
        tmp_hist = cv2.calcHist([img_get], [0], mask_get, [10], [0,180])
        for j in range(10):
            co_list.append(int(tmp_hist[j]))
        for k in range(1,3):
            tmp_hist = cv2.calcHist([img_get], [k], mask_get, [10], [0,255])
            for j in range(10):
                co_list.append(int(tmp_hist[j]))
    else:
        for k in range(3):
            tmp_hist = cv2.calcHist([img_get], [k], mask_get, [10], [0,255])
            for j in range(10):
                co_list.append(int(tmp_hist[j]))

    img_hist = np.array(co_list)
    tmp_hist = np.array(hist_get)
    sim_res_1 = cos_sim(img_hist[0:10], tmp_hist[0:10])
    sim_res_2 = cos_sim(img_hist[10:20], tmp_hist[10:20])
    sim_res_3 = cos_sim(img_hist[20:30], tmp_hist[20:30])
    return [sim_res_1,sim_res_2,sim_res_3]
    

def cos_sim(v_a, v_b):
    a = np.dot(v_a,v_b)
    if a == 0:
        return 0
    return a/(np.linalg.norm(v_a)*np.linalg.norm(v_b))


def deliver_res(res_raw, linemode_thr, hist_thr, hog_sim_thr, mode = 2):
    re_res = []
    for i in res_raw:
        if i[1] < linemode_thr:
            continue
        if mode == 0:
            re_res.append(i)
            continue
        if i[5] < hist_thr or i[6] < hist_thr or i[7] < hist_thr:
            continue
        if mode == 1:
            re_res.append(i)
            continue
        hog_flag = False
        for num,j in enumerate(i[8:]):
            if num%2 == 0:
                if j > 25:
                    hog_flag = True
                    break
            else:
                if j < hog_sim_thr:
                    hog_flag = True
                    break
        if hog_flag:
            continue
        re_res.append(i)
    
    return re_res
    



    



