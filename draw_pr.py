import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


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



def draw_pr(input_pr):
    sor_pr = input_pr[np.argsort(input_pr[:,1])[::-1]]
    #import pdb
    #pdb.set_trace()
    last_rc = 1
    last_pc = 0
    total_area = 0
    pc_points = [0]
    rc_points = [1]
    for i in sor_pr:
        pc_points.append(last_pc)
        rc_points.append(last_rc)
        if i[0] > last_pc:
            pc_points.append(last_pc)
            rc_points.append(i[1])
            last_pc = i[0]
        last_rc = i[1]
    if last_rc > 0:
        total_area += last_pc*last_rc
        pc_points.append(last_pc)
        rc_points.append(0)
    pc_points.append(1)
    rc_points.append(0)
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(rc_points,pc_points)
    plt.plot(sor_pr[:,1],sor_pr[:,0])
    plt.show()




with open('./grad_PR_LM.json', 'r') as f:
    raw_data = json.load(f)


ap_dic = {}
iou_list = [i/100 for i in range(50,100,5)]
for i in range(10):
    ap_list = []
    for j in range(16):
        tmp_pr = get_one_PR(raw_data, i, j, 2)
        if tmp_pr.shape[-1] != 2:
            ap_list.append(0)
        else:
            ap_list.append(get_one_ap(tmp_pr))
    ap_dic[iou_list[i]] = np.average(ap_list)

print(np.average(list(ap_dic.values())))
print(ap_dic)

#tmp_pr = get_one_PR(raw_data, 0, 8, 0)
#draw_pr(tmp_pr)

"""
tar = 'toy_2_1'
img  = cv2.imread("/home/ming/project/bin_picking/data/raw_tmp/"+tar+"/"+tar+".bmp")
with open("/home/ming/project/bin_picking/data/raw_tmp/"+tar+"/grabstr.json", 'r') as f:
    ra = json.load(f)

cv2.circle(img, (ra[0][0], ra[0][1]), 10, (0,255,0), -1, 1)
cv2.circle(img, (ra[1][0], ra[1][1]), 10, (0,255,0), -1, 1)

cv2.imwrite("./res_show/"+tar+".bmp", img)
"""