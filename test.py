# -*- coding: utf-8 -*-  

import os
import cv2
import numpy as np
import json
import argparse
import random

import MvCore



control_arg = argparse.ArgumentParser(prog="CL2D visul")
control_arg.add_argument('-cp','--core_path', help="path to linemodcore.so", type=str, default='./libs/liblinemodcore_60.so')
control_arg.add_argument('-m','--modeling', help="path to modeling", type=str, default='data/train_set_with_label/')
control_arg.add_argument('-s','--scene', help="path to scene", type=str, default='data/test_set/')
control_arg.add_argument('-t','--template', help="path to templates", type=str, default='data/template_data/HSV/')
control_arg.add_argument('-r','--run_mode', help="modeling,detect,show_gt", type=str, default='detect')

args = control_arg.parse_args()

core_obj = MvCore.MvServiceCore(args.core_path)

def vir_raw_info(path_to_info):
    with open(path_to_info+'/normal.json','r') as load_f:
        raw_normal = json.load(load_f)
    with open(path_to_info+'/ob.json','r') as load_f:
        raw_ob = json.load(load_f)
    with open(path_to_info+'/area.json','r') as load_f:
        raw_area = json.load(load_f)
    with open(path_to_info+'/draw.json','r') as load_f:
        raw_draw = json.load(load_f)
    with open(path_to_info+'/grabstr.json','r') as load_f:
        raw_grab = json.load(load_f)
    return raw_normal, raw_ob, raw_area, raw_draw, raw_grab

def save_str_info(obj_get, id_get, path_to_save):
    str_normal, str_ob, str_color, str_area, str_draw, str_grab = obj_get.getTemplate(id_get)
    with open(path_to_save+'/normal.txt','w') as save_f:
        save_f.write(str_normal)
    with open(path_to_save+'/ob.txt','w') as save_f:
        save_f.write(str_ob)
    with open(path_to_save+'/color.txt','w') as save_f:
        save_f.write(str_color)
    with open(path_to_save+'/area.txt','w') as save_f:
        save_f.write(str_area)
    with open(path_to_save+'/draw.txt','w') as save_f:
        save_f.write(str_draw)
    with open(path_to_save+'/grab.txt','w') as save_f:
        save_f.write(str_grab)

def load_str_info(obj_get, mod_id, tmp_id, path_to_load):
    with open(path_to_load+'/normal.txt','r') as save_f:
        a_ = save_f.read()
    with open(path_to_load+'/ob.txt','r') as save_f:
        b_ = save_f.read()
    with open(path_to_load+'/color.txt','r') as save_f:
        c_ = save_f.read()
    with open(path_to_load+'/area.txt','r') as save_f:
        d_ = save_f.read()
    with open(path_to_load+'/draw.txt','r') as save_f:
        e_ = save_f.read()
    with open(path_to_load+'/grab.txt','r') as save_f:
        f_ = save_f.read()
    obj_get.add_Template(mod_id, tmp_id, a_, b_, c_, d_, e_, f_)


def grab_point_draw(tar_res, img, tf_obj):
    pts = tf_obj[tar_res[0]].grab[tar_res[-1]]
    cv2.circle(img, (pts[0][0], pts[0][1]), 10, (0,255,0))
    cv2.circle(img, (pts[1][0], pts[1][1]), 10, (0,255,0))



if args.run_mode == 'modeling':
    souce_dir = args.train#'/home/ming/project/bin_picking/data/raw_tmp/'
    dst_dir = args.orgin#'/home/ming/project/bin_picking/data/build_tmp/'
    all_tmp = os.listdir(souce_dir)
    for i in all_tmp:
        print('Now process:', i) 
        j = i.split('_') 
        model_id = j[0]+'_'+j[1] 
        a_1, b_1, c_, d_, e_, f_ = vir_raw_info(souce_dir+i) 
        print('Build success') 
        img_0 = cv2.imread(souce_dir+i+'/'+i+'.bmp') 
        core_obj.buildTemplate(model_id, i, img_0, a_1, c_, d_, e_, f_)
        if os.path.exists(dst_dir+i):
            for k in os.listdir(dst_dir+i):
                os.remove(dst_dir+i+'/'+k)
            save_str_info(core_obj, i, dst_dir+i)
        else:
            os.mkdir(dst_dir+i)
            save_str_info(core_obj, i, dst_dir+i)
            print('Save success')





else:
    if os.path.exists('./color_dic.json'):
        with open('./color_dic.json', 'r') as f:
            color_list = json.load(f)
    else:
        color_list = []
        for i in range(20):
            color_list.append((random.randrange(255), random.randrange(255), random.randrange(255)))
        with open('./color_dic.json', 'w') as f:
            json.dump(color_list, f)

    if not os.path.exists('./res_show/'):
        os.mkdir('./res_show/')

    souce_dir = args.template
    all_tmp = os.listdir(souce_dir)
    total_range = []
    for i in all_tmp:
        j = i.split('_')
        model_id = j[0]+'_'+j[1]
        load_str_info(core_obj, model_id, i, souce_dir+i)
        total_range.append(model_id)
    total_range = list(set(total_range))
    core_obj.set_range(total_range)
    font = cv2.FONT_HERSHEY_SIMPLEX

    scene_dir = args.scene

if args.run_mode == 'detect':
    import time
    for i in os.listdir(scene_dir):
        if i[-3:] == 'bmp':
            print(i)
            img_0  = cv2.imread(scene_dir+i)
            img_back = np.zeros(img_0.shape, np.uint8)
            img_canny = cv2.Canny(cv2.GaussianBlur(img_0, (3,3), 0,0), 130, 190)
            img_edges_back = np.zeros(img_0.shape, np.uint8)
            img_edges_back[:,:,0] = img_canny
            t1 = time.time()
            res = core_obj.match_a_img(img_0)
            print("\n#DEBUG total time:", time.time()-t1)
            if res != []:
                for j in res:
                    curr_color = color_list[int(j[0].split('_')[1])-1]
                    curr_polygon = np.array([[[k[0]+j[2], k[1]+j[3]] for k in core_obj.tf[j[0]].area[j[4]]]])
                    curr_rect = cv2.boundingRect(curr_polygon)
                    cv2.fillPoly(img_back, curr_polygon, curr_color)
                    cv2.putText(img_0, j[0]+' '+str(j[1]), (j[2], j[3]), font, 1.2, curr_color, 3)
                    cv2.putText(img_back, j[0]+' '+str(j[1]), (j[2], j[3]), font, 1.2, curr_color, 3)
                    cv2.rectangle(img_back, curr_rect[:2], (curr_rect[0]+curr_rect[2],curr_rect[1]+curr_rect[3]),curr_color, 3)
                    cv2.rectangle(img_0, curr_rect[:2], (curr_rect[0]+curr_rect[2],curr_rect[1]+curr_rect[3]),curr_color, 3)
            end_img = cv2.addWeighted(img_0, 0.5, img_back, 0.5, 0)
            cv2.imwrite('./res_show/'+i, end_img)
            print('process', i)
    print(MvCore.TIME_LIST)

elif args.run_mode == 'show_gt':
    for k in os.listdir(scene_dir):
        if k[-3:] == 'bmp':
            IMG = cv2.imread(scene_dir+k)
            BACK = np.zeros(IMG.shape, np.uint8)
            with open(scene_dir+k[:-3]+'json') as f:
                JSON_GT = json.load(f)
            for i in JSON_GT['shapes']:
                    if i['label'] != 'nan':
                        curr_toy_id = i['label'].split('_')[1]
                        curr_polygon = np.array([[[int(j[0]), int(j[1])] for j in i['points']]])
                        curr_rect = cv2.boundingRect(curr_polygon)
                        cv2.putText(BACK, i['label'], (int(curr_rect[0]+curr_rect[2]/2), int(curr_rect[1]+curr_rect[3]/2)), font, 1.2, color_list[int(curr_toy_id)-1], 3)
                        cv2.putText(IMG, i['label'], (int(curr_rect[0]+curr_rect[2]/2), int(curr_rect[1]+curr_rect[3]/2)), font, 1.2, color_list[int(curr_toy_id)-1], 3)
                        cv2.fillPoly(BACK, curr_polygon, color_list[int(curr_toy_id)-1])
                        cv2.rectangle(BACK, curr_rect[:2], (curr_rect[0]+curr_rect[2],curr_rect[1]+curr_rect[3]),color_list[int(curr_toy_id)-1], 3)
                        cv2.rectangle(IMG, curr_rect[:2], (curr_rect[0]+curr_rect[2],curr_rect[1]+curr_rect[3]),color_list[int(curr_toy_id)-1], 3)
            FUSION = cv2.addWeighted(IMG, 0.5, BACK, 0.5, 0)
            cv2.imwrite('./res_show/'+k, FUSION)