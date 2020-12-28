# -*- coding: utf-8 -*-  

import cv2
import numpy as np
import json
import copy

PAD_L = 7
def yaml_rotate_process(get_img, get_index, rot_angle, calculator):
    """
    Get fea information of current degree, small img normal fea only
    @param:
        get_img: source img get
        get_index: position of where normal fea(small)
        rot_angle: current rot degree
        calculator: linemod object
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = get_img.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -rot_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rot_raw = cv2.warpAffine(get_img, M, (nW, nH))
    rot_img = cv2.copyMakeBorder(rot_raw, PAD_L,PAD_L,PAD_L,PAD_L, cv2.BORDER_REPLICATE)
    tl_x, tl_y = rot_img.shape[1]//2, rot_img.shape[0]//2 

    fea_points = []

    for i in get_index:
        tmp_point = np.array([i[0], i[1]])
        re_1 = M[0:,:2].dot(tmp_point.T)
        re_2 = re_1 + M[0:,-1]
        re_2 += PAD_L
        tmp_part = calculator.cal_quant(rot_img[int(re_2[1])-2:int(re_2[1])+3, int(re_2[0])-2:int(re_2[0])+3])[1:-1,1:-1]
        tmp_array =np.array([0,0,0,0,0,0,0,0])
        for j in range(8):
            tmp_array[j] = np.sum(tmp_part == j)
        fea_points.append([int(re_2[0]), int(re_2[1]), int(tmp_array.argmax())])
    re_list = [rot_img.shape[:2], [int(tl_x), int(tl_y)], fea_points]
    del rot_img
    return re_list

def many_rotate_process(get_img, get_index, get_index_ob, get_index_area, get_index_grab, rot_angle, calculator):
    """
    Get fea information of current degree
    @param:
        get_img: source img
        get_index: position of normal fea
        get_index_ob: position of ob fea
        get_index_area: base polygon 
        get_index_grab: position of grab point(or draw point)
        rot_angle: current rot degree
        calculator: linemod object
    @return:
        re_list: normal fea of current degree
        re_list_ob: ob fea of current degree
        re_list_area: area polygon of current degree
        re_list_grab: grab point of current degree
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = get_img.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -rot_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rot_raw = cv2.warpAffine(get_img, M, (nW, nH))
    rot_img = cv2.copyMakeBorder(rot_raw, PAD_L,PAD_L,PAD_L,PAD_L, cv2.BORDER_REPLICATE)
    tl_x, tl_y = rot_img.shape[1]//2, rot_img.shape[0]//2 

    #1. get rot normal feapoint
    fea_points = []
    for i in get_index:
        tmp_point = np.array([i[0], i[1]])
        re_1 = M[0:,:2].dot(tmp_point.T)
        re_2 = re_1 + M[0:,-1]
        re_2 += PAD_L
        tmp_part = calculator.cal_quant(rot_img[int(re_2[1])-2:int(re_2[1])+3, int(re_2[0])-2:int(re_2[0])+3])[1:-1,1:-1]
        tmp_array =np.array([0,0,0,0,0,0,0,0])
        for j in range(8):
            tmp_array[j] = np.sum(tmp_part == j)
        fea_points.append([int(re_2[0]), int(re_2[1]), int(tmp_array.argmax())])


    #2. get rot ob position
    re_list_ob_tar = []
    for i in get_index_ob:
        tmp_point = np.array([i[0]+2, i[1]+2])
        re_1 = M[0:,:2].dot(tmp_point.T)
        re_2 = re_1 + M[0:,-1]
        re_2 += PAD_L
        re_list_ob_tar.append([int(re_2[0])-tl_x, int(re_2[1])-tl_y, int(tmp_array.argmax())])

    #3. get rot area polygon
    re_list_area =[]
    for i in get_index_area:
        tmp_point = np.array([i[0]+2, i[1]+2])
        re_1 = M[0:,:2].dot(tmp_point.T)
        re_2 = re_1 + M[0:,-1]
        re_2 += PAD_L
        re_list_area.append([int(re_2[0])-tl_x, int(re_2[1])-tl_y])

    #4. get ob HOG fea
    re_list_ob = hog_fea_process(rot_img, tl_x, tl_y, re_list_ob_tar, calculator) 

    #5. get rot grab point
    re_list_grab = []
    for i in get_index_grab:
        tmp_point = np.array([i[0]+2, i[1]+2])
        re_1 = M[0:,:2].dot(tmp_point.T)
        re_2 = re_1 + M[0:,-1]
        re_2 += PAD_L
        re_list_grab.append([int(re_2[0])-tl_x, int(re_2[1])-tl_y])

    re_list = [rot_img.shape[:2], [int(tl_x), int(tl_y)], fea_points]
    del rot_img
    return re_list, re_list_ob, re_list_area, re_list_grab

def get_quant_ang(img_in):
    im_guass = cv2.GaussianBlur(img_in , (7,7), 0, 0, cv2.BORDER_REFLECT )#高斯模糊
    sobel_3dx = cv2.Sobel(im_guass, 3, 1, 0, ksize = 3, scale = 1, borderType = cv2.BORDER_REFLECT )#x方向导数
    sobel_3dy = cv2.Sobel(im_guass, 3, 0, 1, ksize = 3, scale = 1, borderType = cv2.BORDER_REFLECT )#y方向导数
    mag_all = np.multiply(sobel_3dx.astype(np.int32), sobel_3dx.astype(np.int32)) + np.multiply(sobel_3dy.astype(np.int32), sobel_3dy.astype(np.int32))#梯度模值的平方
    mag_index = np.argmax(mag_all, axis = 2)#模值最大在哪个通道
    mag = np.max(mag_all, axis = 2)#获得模值最大的梯度图
    sobel_dx = sobel_3dx[:,:,0].copy()
    sobel_dy = sobel_3dx[:,:,0].copy()
    for i in range(sobel_dx.shape[0]):#获得相应的x和y方向导数
        for j in range(sobel_dx.shape[1]):
            sobel_dx[i,j] = sobel_3dx[i,j,mag_index[i,j]]
            sobel_dy[i,j] = sobel_3dy[i,j,mag_index[i,j]]
    sobel_ag = np.zeros(mag.shape,dtype = np.float32)
    cv2.phase(sobel_dx.astype(np.float32) , sobel_dy.astype(np.float32) ,  sobel_ag, angleInDegrees = True)#获得相应的0到360度方向
    ang_16 = np.multiply(sobel_ag, 16/360).astype(np.uint8)#量化为8个方向
    ang_8 = ang_16 & 7
    return ang_8,mag
def get_quant_mask(get_ag, get_mag ):#获得初始的mask
    tmp_mask = np.ones(get_ag.shape, np.uint8)
    re_mask = np.zeros(get_ag.shape, np.uint8 )
    for i in range(1, get_ag.shape[0]-1):
        for j in range(1, get_ag.shape[1]-1):
            for k in range(8):
                if tmp_mask[i,j] == 1:
                    if get_mag[i,j] >= 50 and get_mag[i,j] == get_mag[i-1:i+2, j-1:j+2].max():
                        for k in range(8):
                            if np.sum(get_ag[i-1:i+2, j-1:j+2] == k) >= 4 and get_ag[i,j] == k:
                                tmp_mask[i-10:i+11, j-10:j+11] = 0
                                re_mask[i,j] = 255
                                break
    idx_list = np.where(re_mask == 255)
    re_list = []
    for i in range(len(idx_list[0])):
        re_list.append([idx_list[1][i], idx_list[0][i]])
    return re_list
def normal_mag(get_mag):#归一化mag图
    tmp_max = np.max(get_mag)
    tmp_min = np.min(get_mag)
    good_mag = get_mag.astype(np.int64)
    return (np.multiply((good_mag-tmp_min)/(tmp_max-tmp_min), 255)).astype(np.uint8)
def creat_yaml_model(model_id, templates_list, templates_list_small):
    tar_file = ''
    tar_file += ('%YAML:1.0\n---\nclass_id: "'+model_id+'"\npyramid_levels: 2\ntemplate_pyramids:\n')
    for i in range(len(templates_list)):
        tar_file += ('   -\n')
        tar_file += ('      template_id: '+str(i)+'\n')
        tar_file += ('      templates:\n         -\n')
        tar_file += ('            width: '+str(templates_list[i][0][1])+'\n')
        tar_file += ('            height: '+str(templates_list[i][0][0])+'\n')
        tar_file += ('            tl_x: '+str(templates_list[i][1][0])+'\n')
        tar_file += ('            tl_y: '+str(templates_list[i][1][1])+'\n')
        tar_file += ('            pyramid_level: 0\n            features:\n')
        for j in range(len(templates_list[i][2])):
            single_points = templates_list[i][2][j]
            tar_file += ('               - [ '+str(single_points[0])+', '+str(single_points[1])+', '+str(single_points[2])+' ]\n')
        tar_file += ('         -\n')
        tar_file += ('            width: '+str(templates_list_small[i][0][1])+'\n')
        tar_file += ('            height: '+str(templates_list_small[i][0][0])+'\n')
        tar_file += ('            tl_x: '+str(templates_list_small[i][1][0])+'\n')
        tar_file += ('            tl_y: '+str(templates_list_small[i][1][1])+'\n')
        tar_file += ('            pyramid_level: 1\n            features:\n')
        for j in range(len(templates_list_small[i][2])):
            single_points = templates_list_small[i][2][j]
            tar_file += ('               - [ '+str(single_points[0])+', '+str(single_points[1])+', '+str(single_points[2])+' ]\n')
    return tar_file

def auto_gen_templates(cal_obj,template_id, source_img_get, source_get_normal, source_part_list, source_catch_point_draw, source_catch_point_grab, area_mask_get, degree_step = 1):
    """
    Use This Function to gen a normal template
    @param:
        cal_obj: instance of loaded linmodCore
        template_id: the id of target template
        source_img_get: the BGR img of target template
        source_get_normal: a List of target template
        source_part_list: a List of target template's Obvious feature points
        source_color_list: a List of target template's Color position
        source_catch_point_draw: a List of target template's Draw position
        source_catch_point_grab: a List of target template's Grab position
        area_mask_get: a List of a polygon, mark the area of template
    @return:
        yamlstr: String of normal templates
        partstr: String of Obvious templates
        colorstr: String of Color templates
        areastr: String of area
        drawstr: String of draw point
        grabstr: String of grab point
    """
    print("开始模板生成")
    print("template id:", template_id)
        
    #papare some information
    #1.if there no grab point use draw information,else use grab information
    if len(source_catch_point_grab) == 0:
        catch_source = source_catch_point_draw
    else:
        catch_source = source_catch_point_grab
    #2.need two img of two scale for linemod Hierarchical features (two pyramid level)
    base_normal = copy.deepcopy(source_get_normal)
    base_img = source_img_get.copy() 
    base_img_small = cv2.resize(base_img, (base_img.shape[1]//2, base_img.shape[0]//2))
    #3.feature for level 2 creat by hand, but for level 2 ,its auto get
    sour_ag,sour_mag = get_quant_ang(base_img_small)
    base_normal_small = get_quant_mask(sour_ag, normal_mag(sour_mag))
    #4. some empty list
    nm_list = []
    nm_sm_list = []
    ob_list =[]
    ar_list = []
    ca_list = []
    #5. get template of 360 degree
    for i in range(0,360,degree_step):
        print('\r', end = '')
        a_, b_, d_, e_ = many_rotate_process(base_img, base_normal, source_part_list, area_mask_get, catch_source, i, cal_obj) 
        nm_list.append(a_)
        nm_sm_list.append(yaml_rotate_process(base_img_small, base_normal_small, i, cal_obj))
        ob_list.append(b_)
        ar_list.append(d_)
        ca_list.append(e_)
        print(i, end = '')
    print("\n", end ='')
    #6. encode normal feature into opencv format
    yamlstr = creat_yaml_model(template_id, nm_list, nm_sm_list)
    if len(source_catch_point_grab) == 0:
        gr_list = []
        dr_list = ca_list
    else:
        dr_list = []
        gr_list = ca_list
    #7. calculate color hist info
    hist_mask = np.zeros(base_img.shape[:-1], np.uint8)
    area_np = np.array([[[int(k[0]),int(k[1])] for k in area_mask_get]])
    cv2.fillPoly(hist_mask, area_np, 255)
    hsv_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    co_list = []
    tmp_hist = cv2.calcHist([hsv_img], [0], hist_mask, [10], [0,180])
    for j in range(10):
        co_list.append(int(tmp_hist[j]))
    for k in range(1,3):
        tmp_hist = cv2.calcHist([hsv_img], [k], hist_mask, [10], [0,255])
        for j in range(10):
            co_list.append(int(tmp_hist[j]))
    return yamlstr, json.dumps(ob_list), json.dumps(co_list), json.dumps(ar_list), json.dumps(dr_list), json.dumps(gr_list)

def hog_fea_process(rot_img, cen_x, cen_y, ob_tar, cal_):
    """
    This Function to get HOG feature
    @param:
        rot_img: img of current degree
        cen_x: center coordinate x 
        cen_y: center coordinate y
        ob_tar: position of where hog fea need to get
        cal_: linemod object
    """
    blur_img = cv2.GaussianBlur(rot_img, (3,3), 0,0)
    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1,0, ksize = 3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0,1, ksize = 3)
    img_mag = cv2.magnitude(sobel_x, sobel_y)
    img_mag[img_mag >255] = 255
    img_norm = img_mag/255

    re_list_hog = []

    for i in ob_tar:
        tmp_pos_x = cen_x + i[0]
        tmp_pos_y = cen_y + i[1]
        ang_patch = cal_.cal_quant(rot_img[tmp_pos_y-8:tmp_pos_y+9, tmp_pos_x-8:tmp_pos_x+9])[1:-1,1:-1]
        tmp_hog = []
        for j in range(8):
            tmp_hog.append(np.sum(img_norm[tmp_pos_y-7:tmp_pos_y+8, tmp_pos_x-7:tmp_pos_x+8][np.where(ang_patch == j)]))
        re_list_hog.append([i[0], i[1], tmp_hog])

    return re_list_hog
