
import json 
import cv2
import numpy as np
import copy
import random
import json
import time
import os
import argparse
from shapely.geometry import Polygon
import multiprocessing

import MvCore


control_arg = argparse.ArgumentParser(prog="Mings Alg vir playground", description="train, run, and meaure the alg")
control_arg.add_argument('-cp','--core_path', help="path to linemodcore.so", type=str, default='./libs/liblinemodcore_50.so')
control_arg.add_argument('-s','--scene', help="path to scene", type=str, default='data/test_set/')
control_arg.add_argument('-t','--template', help="path to templates", type=str, default='data/template_data/')
control_arg.add_argument('-r','--result', help="reault", type=str, default='./raw_ras_save/')
control_arg.add_argument('-m','--mode', help="MODE: HSV RGB LM", type=str, default='HSV')

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

def rot_img_fuse(info_get, scene_get, patch_get, img_mother):
    (h, w) = patch_get.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -info_get[4], 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rot_raw = cv2.warpAffine(patch_get, M, (nW, nH))
    tl_x, tl_y = rot_raw.shape[1]//2, rot_raw.shape[0]//2 
    rot_raw = cv2.GaussianBlur(rot_raw, (3,3), 0,0)
    #rot_raw = cv2.bilateralFilter(rot_raw,5, 20, 5)
    patch_edges_1 = cv2.Canny(rot_raw, 130, 180)
    patch_edges = patch_edges_1#cv2.dilate(patch_edges_1, np.ones((3,3),np.uint8), iterations = 1)
    cen_x, cen_y = info_get[2], info_get[3]
    scene_get[cen_y-tl_y:patch_edges.shape[0]+(cen_y-tl_y), cen_x-tl_x:patch_edges.shape[1]+(cen_x-tl_x), 2] = cv2.bitwise_or(patch_edges, scene_get[cen_y-tl_y:patch_edges.shape[0]+(cen_y-tl_y), cen_x-tl_x:patch_edges.shape[1]+(cen_x-tl_x), 2])
    code_num = str(random.randint(0,100000))
    cv2.imwrite('./img_pair_2/'+code_num+'_a.bmp', img_mother[cen_y-tl_y:patch_edges.shape[0]+(cen_y-tl_y), cen_x-tl_x:patch_edges.shape[1]+(cen_x-tl_x)])
    cv2.imwrite('./img_pair_2/'+code_num+'_b.bmp', rot_raw)



def iou(rect1,rect2):
    '''
    计算两个矩形的交并比
    :param rect1:第一个矩形框。表示为x,y,w,h，其中x,y表示矩形右上角的坐标
    :param rect2:第二个矩形框。
    :return:返回交并比，也就是交集比并集
    '''
    x1,y1,w1,h1=rect1
    x2,y2,w2,h2=rect2

    inter_w=(w1+w2)-(max(x1+w1,x2+w2)-min(x1,x2))
    inter_h=(h1+h2)-(max(y1+h1,y2+h2)-min(y1,y2))

    if inter_h<=0 or inter_w<=0:#代表相交区域面积为0
        return 0
    #往下进行应该inter 和 union都是正值
    inter=inter_w * inter_h

    union=w1*h1+w2*h2-inter
    return inter/union

def iou_mask(poly_1, poly_2):
    if poly_1.intersects(poly_2):
        ins_a = poly_1.intersection(poly_2).area
        uni_a = poly_1.union(poly_2).area
        return ins_a/uni_a
    else:
        return 0
    

def get_one_PR(in_data, idx_iou, idx_cg, idx_bm):
    """
    @idx_iou: which iou chose [0,1,...,9] -> [.5,.55,...,.95] 
    @idx_cg: which category chose [0,1,..,15] -> [toy_1,toy_2,...,toy_16]
    @idx_bm: which PR chose [0,2] -> [bbox, mask]
    """
    re_list = []
    for i in in_data:
        tmp = [i[idx_iou][idx_cg][idx_bm],i[idx_iou][idx_cg][idx_bm+1]]
        if tmp != [0,0]:
            re_list.append([i[idx_iou][idx_cg][idx_bm],i[idx_iou][idx_cg][idx_bm+1]])
    return np.array(re_list)

def get_one_ap(input_pr):
    sor_pr = input_pr[np.argsort(input_pr[:,1])[::-1]]
    last_rc = 1
    last_pc = 0
    total_area = 0
    for i in sor_pr:
        total_area += last_pc * (last_rc - i[1])
        if i[0] > last_pc:
            last_pc = i[0]
        last_rc = i[1]
    if last_rc > 0:
        total_area += last_pc*last_rc
    return total_area


class img_info():
    def __init__(self, img_get, json_get):
        self.empty_back = np.zeros(img_get.shape[:-1],np.uint8)
        self.all_toy_dic = {}
        self.all_toy_dic_bbox = {}

        self.toy_num = {}
        self.recall_dic_mask = {}
        self.iou_dic_re_mask = {}
        self.recall_dic_bbox = {}
        self.iou_dic_re_bbox = {}

        self.predict_num = {}
        self.precision_dic_mask = {}
        self.iou_dic_pr_mask = {}
        self.precision_dic_bbox = {}
        self.iou_dic_pr_bbox = {}

        for i in json_get['shapes']:
            if i['label'] != 'nan':
                curr_toy_id = i['label'].split('_')[1]
                #draw_background = copy.deepcopy(self.empty_back)
                curr_polygon = np.array([[[int(j[0]), int(j[1])] for j in i['points']]])
                #cv2.fillPoly(draw_background, curr_polygon, 255)
                if curr_toy_id not in self.all_toy_dic.keys():
                    self.all_toy_dic[curr_toy_id] = []
                    self.all_toy_dic_bbox[curr_toy_id] = []
                    self.toy_num[curr_toy_id] = 1
                else:
                    self.toy_num[curr_toy_id] += 1
                self.all_toy_dic[curr_toy_id].append(Polygon(curr_polygon[0]))
                self.all_toy_dic_bbox[curr_toy_id].append(cv2.boundingRect(curr_polygon))
        for i in self.all_toy_dic.keys():
            self.iou_dic_re_bbox[i] = [0]*len(self.all_toy_dic[i])
            self.iou_dic_re_mask[i] = [0]*len(self.all_toy_dic[i])
    def measure_res(self, res_get, tf_get):
        self.recall_dic_mask = {}
        self.recall_dic_bbox = {}
        for i in self.all_toy_dic.keys():
            self.iou_dic_re_bbox[i] = [0]*len(self.all_toy_dic[i])
            self.iou_dic_re_mask[i] = [0]*len(self.all_toy_dic[i])

        self.predict_num = {}
        self.precision_dic_mask = {}
        self.iou_dic_pr_mask = {}
        self.precision_dic_bbox = {}
        for i in res_get:
            curr_id = i[0].split('_')[1]
            curr_polygon = np.array([[[k[0]+i[2], k[1]+i[3]] for k in tf_get[i[0]].area[i[4]]]])
            curr_rect = cv2.boundingRect(curr_polygon)
            #curr_draw = copy.deepcopy(self.empty_back)
            #cv2.fillPoly(curr_draw, curr_polygon, 255)
            if curr_id in self.predict_num.keys():
                self.predict_num[curr_id] += 1
            else:
                self.predict_num[curr_id] = 1
                self.iou_dic_pr_bbox[curr_id] = []
                self.iou_dic_pr_mask[curr_id] = []
            self.iou_dic_pr_mask[curr_id].append(0)
            self.iou_dic_pr_bbox[curr_id].append(0)
            if curr_id in self.all_toy_dic.keys():
                for nm,j in enumerate(self.all_toy_dic[curr_id]):
                    """
                    inser = cv2.bitwise_and(curr_draw, j)
                    inser_area = np.sum(inser > 0)
                    if inser_area > 0:
                        uni = cv2.bitwise_or(curr_draw, j)
                        uni_area = np.sum(uni > 0)
                        rate = inser_area/uni_area
                    else:
                        rate = 0
                    """
                    rate = iou_mask(j, Polygon(curr_polygon[0]))
                    if rate > self.iou_dic_pr_mask[curr_id][-1]:
                        self.iou_dic_pr_mask[curr_id][-1] = rate
                    if rate > self.iou_dic_re_mask[curr_id][nm]:
                        self.iou_dic_re_mask[curr_id][nm] = rate

                    rate = iou(curr_rect,self.all_toy_dic_bbox[curr_id][nm])
                    if rate > self.iou_dic_pr_bbox[curr_id][-1]:
                        self.iou_dic_pr_bbox[curr_id][-1] = rate
                    if rate > self.iou_dic_re_bbox[curr_id][nm]:
                        self.iou_dic_re_bbox[curr_id][nm] = rate
        return self.iou_dic_re_mask,self.iou_dic_re_bbox,self.predict_num,self.iou_dic_pr_mask,self.iou_dic_pr_bbox
                    
                    
    def cal_PR(self, iou_rate):
        for i in self.toy_num.keys():
            self.recall_dic_mask[i] = np.sum(np.array(self.iou_dic_re_mask[i]) >= iou_rate)
            self.recall_dic_bbox[i] = np.sum(np.array(self.iou_dic_re_bbox[i]) >= iou_rate)
        
        for i in self.predict_num.keys():
            self.precision_dic_mask[i] = np.sum(np.array(self.iou_dic_pr_mask[i]) >= iou_rate)
            self.precision_dic_bbox[i] = np.sum(np.array(self.iou_dic_pr_bbox[i]) >= iou_rate)

def pall_pr(arg_pack):
    get_res, get_info, get_tf, j, rs_k, md= arg_pack
    grad_point = EvDector.deliver_res(get_res, j[0], j[1]/100, j[2]/100, md)
    a,b,c,d,e = get_info.measure_res(grad_point,get_tf)
    #get_info.measure_res(grad_point,get_tf)
    #get_info.cal_PR(iou_r/100)
    return rs_k,a,b,c,d,e

def get_res(mode = 'HSV'):
    if(not os.path.exists(args.result)):
        os.mkdir(args.result)
    count = 0
    for i in os.listdir(scene_dir):
        if i[-3:] == 'bmp':
            count += 1
            img_0  = cv2.imread(scene_dir+i)
            t1 = time.time()
            print("\n#>Processing IMG:", i, str(count)+'/30')
            save_res = core_obj.papare_res(img_0, mode)
            print("#>DONE", time.time()-t1)
            with open(args.result+i[:-4]+'.json', 'w') as f:
                json.dump(save_res, f)
    print("\n#> DONE !")


color_list = []
for i in range(20):
    color_list.append((random.randrange(255), random.randrange(255), random.randrange(255)))
if args.mode == 'HSV':
    souce_dir = args.template+'HSV/'
else:
    souce_dir = args.template+'RGB/'
all_tmp = os.listdir(souce_dir)
total_range = []
for i in all_tmp:
    j = i.split('_')
    model_id = j[0]+'_'+j[1]
    load_str_info(core_obj, model_id, i, souce_dir+i)
    total_range.append(model_id)
#debug
total_range = list(set(total_range))
#total_range = ['toy_11']
core_obj.set_range(total_range)
font = cv2.FONT_HERSHEY_SIMPLEX
scene_dir = args.scene

get_res(args.mode)
img_infos = {}
import EvDector
for i in os.listdir(scene_dir):
    if i[-3:] == 'bmp':
        img = cv2.imread(scene_dir+i)
        with open(scene_dir+i[:-4]+'.json') as f:
            jstr = json.load(f)
        img_infos[i[:-4]] = img_info(img, jstr)
        

raw_ress = {}
for i in os.listdir(args.result):
    with open(args.result+i) as f:
        raw_res = json.load(f)
    raw_ress[i[:-5]] = raw_res


if args.mode == 'LM':
    process_mode = 0
    grad_thr = [[i, 0, 0] for i in range(50,100)]
else:
    process_mode = 1
    grad_thr = []
    for a in range(50, 100):
        for b in range(50, 100):
            grad_thr.append([a,b,0])

points_grad = []
time_cout = 0
last_time = 0
print(">>>>> MODE:", process_mode)
for c_n,j in enumerate(grad_thr):
    print("\r###", c_n, '/', len(grad_thr), "elp tim:"+str(time_cout/60), "eta:"+str((len(grad_thr)-c_n)*last_time/60), "last:", last_time, j, end = '')
    t1 = time.time()
    itmes = [(raw_ress[rs_k],img_infos[rs_k],core_obj.tf, j, rs_k, process_mode) for rs_k in raw_ress.keys()]
    p = multiprocessing.Pool(6)
    b = p.map(pall_pr, itmes)
    p.close()
    p.join()
    for tmp_k,a,b,c,d,e in b:
        img_infos[tmp_k].iou_dic_re_mask = a
        img_infos[tmp_k].iou_dic_re_bbox = b
        img_infos[tmp_k].predict_num = c
        img_infos[tmp_k].iou_dic_pr_mask = d
        img_infos[tmp_k].iou_dic_pr_bbox = e
        #img_infos[tmp_k].cal_PR(iou_r/100)
    last_time = (time.time()-t1)
    time_cout+=last_time
    #print("cal_get", time.time()-t1)

    points_iou = []
    for iou_r in range(50,100,5):
        points_class = []
        for tmp_key in img_infos.keys():
            img_infos[tmp_key].cal_PR(iou_r/100)
        for i in range(1,17):
            total_gt_num = 0
            total_pd_num = 0
            total_rc_bbox = 0
            total_rc_mask = 0
            total_pc_bbox = 0
            total_pc_mask = 0
            for rs_k in raw_ress.keys():
                if str(i) in img_infos[rs_k].toy_num.keys():
                    total_gt_num += img_infos[rs_k].toy_num[str(i)]
                    total_rc_bbox += img_infos[rs_k].recall_dic_bbox[str(i)]
                    total_rc_mask += img_infos[rs_k].recall_dic_mask[str(i)]
                if str(i) in img_infos[rs_k].predict_num.keys():
                    total_pd_num += img_infos[rs_k].predict_num[str(i)]
                    total_pc_bbox += img_infos[rs_k].precision_dic_bbox[str(i)]
                    total_pc_mask += img_infos[rs_k].precision_dic_mask[str(i)]

            if total_pd_num != 0:
                tmp_rate = [total_pc_bbox/total_pd_num, total_rc_bbox/total_gt_num, total_pc_mask/total_pd_num, total_rc_mask/total_gt_num]
                if min(tmp_rate) < 0 or max(tmp_rate) > 1:
                    import pdb
                    pdb.set_trace()
                points_class.append(tmp_rate)
            else:
                points_class.append([0,0,0,0])

        points_iou.append(points_class)
    points_grad.append(points_iou)

with open('grad_PR_'+str(args.mode)+'.json', 'w') as f:
    json.dump(points_grad, f)
    
"""    
with open('./grad_PR_RGB.json', 'r') as f:
    points_grad = json.load(f)
"""


def print_AP(raw_data, mode="Bounding BOX"):
    print('\n==========',mode, 'AP','===========')
    if mode == "Bounding BOX":
        mcd = 0
    else:
        mcd = 2
    ap_dic = {}
    iou_list = [i/100 for i in range(50,100,5)]
    for i in range(10):
        ap_list = []
        for j in range(16):
            tmp_pr = get_one_PR(raw_data, i, j, mcd)
            if tmp_pr.shape[-1] != 2:
                ap_list.append(0)
            else:
                ap_list.append(get_one_ap(tmp_pr))
        ap_dic[iou_list[i]] = np.average(ap_list)

    print("# AP:", np.average(list(ap_dic.values())))
    for i in ap_dic.keys():
        print("AP@"+str(i)+':', ap_dic[i], end='||')
    print('')

print_AP(points_grad, "Bounding BOX")
print_AP(points_grad, "Mask")
            
        
            

    



        


