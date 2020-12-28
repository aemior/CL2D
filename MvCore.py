# -*- coding: utf-8 -*-  
import base64
import time
import numpy as np
import cv2
import six


import LineModWapper as LineModWapper
import MvDector as MvDector
import TemplateBuilder as TemplateBuilder
import TemplateFactory as TemplateFactory

TIME_LIST = []

class MvServiceCore(object):
    def __init__(self, path_get):
        self.so_path = path_get#rhopy.store['plugins']['linemod_so_path']
        print("视觉核心启动")
        try:
            self.lm = LineModWapper.linemodcore(self.so_path)
        except Exception as e:
            print("动态库 LineModCore.so 加载失败...: {}".format(e))
        else:
            print("动态库 LineModCore.so 加载成功")
        self.tf = TemplateFactory.TemplateFactory()
        self.tf_sync_flag = False
        self.range_list = []

    def buildTemplate(self, modelId, templateId, template_im, normal_get, obvious_get, area_get, draw_get, grab_get, degree_step = 1):
        """
        Build a Template
        @param:
            modelId: model ID template belong to
            templateId: template ID
            template_im: template Image Input
            normal_get: the list of raw normal feature point position
            obvious_get: the list of raw obvious feature point position
            area_get: the list of raw polygon mark
            draw_get: the list of raw draw point position
            grab_get: the list of raw garb point position
        """
        yamlstr, partstr, colorstr, areastr, drawstr, grabstr = TemplateBuilder.auto_gen_templates(self.lm, templateId, template_im, normal_get, obvious_get, draw_get, grab_get, area_get, degree_step)
        tmp_template = TemplateFactory.atom_template(templateId)
        a_, b_, c_, d_, e_ = TemplateFactory.DecodeTemplate(partstr, colorstr, areastr, drawstr, grabstr)
        tmp_template.fill(yamlstr, a_, b_, c_, d_, e_)
        if self.tf.has_template(templateId):
            self.tf.reset_template(tmp_template)
        else:
            self.tf.add_template(modelId, tmp_template)
        self.tf_sync_flag = False
        print("生成模板:"+templateId)

    def getTemplate(self,templateId):
        """
        Get template from template factory
        return is Encoding to string
        @param:
            templateId: the templateId
        @return:
            normal_str, obvious_str, color_str, area_str, draw_str, grab_str
        """
        if self.tf.has_template(templateId):
            partstr, colorstr, areastr, drawstr, grabstr = TemplateFactory.EncodeTemplate(self.tf[templateId].obvious, self.tf[templateId].color, self.tf[templateId].area, self.tf[templateId].draw, self.tf[templateId].grab)
            return self.tf.get_normal(templateId), partstr, colorstr, areastr, drawstr, grabstr
        else:
            print("获取模板失败 ID:"+templateId)
            raise KeyError

    def buildSpeTemplate(self, tep_id_a, tep_id_b, img_a, img_b, normal_get_a, normal_get_b, ob_get_a, ob_get_b):
        """
        Build a Specific template
        @param:
            tep_id_a: ID of template A
            tep_id_b: ID of template B
            img_a: Image of template A
            img_b: Image of template B
            normal_get_a: List of template A normal
            normal_get_b: List of template B normal
            ob_get_a: List of template A Obvious
            ob_get_b: List of template B Obvious
        """
        yaml_a,  yaml_b, ob_str_a, ob_str_b = TemplateBuilder.auto_get_templates_spe(self.lm, tep_id_a, tep_id_b, img_a, img_b,  normal_get_a, normal_get_b, ob_get_a, ob_get_b)
        tmp_spetemplate = TemplateFactory.spe_template(tep_id_a, tep_id_b)
        a_, b_ = TemplateFactory.DecodeSpeTemplate(ob_str_a, ob_str_b)
        tmp_spetemplate.fill(yaml_a, yaml_b, a_, b_)
        self.tf.add_sep(tmp_spetemplate)
        self.tf_sync_flag = False
        print("build spe template:"+tep_id_a+tep_id_b)

    def getSpeTemplate(self, tep_id_a, tep_id_b):
        """
        Get Specific template from template factory
        Return is Encoding to string
        @param:
            tep_id_a: the Template ID A
            tep_id_b: the Template ID B
        @return:
            normal_str_a, normal_str_b, obvious_str_a, obvious_str_b
        """
        if self.tf.has_spe(tep_id_a, tep_id_b):
            ob_a_str, ob_b_str = TemplateFactory.EncodeSpeTemplate(self.tf.get_spe_ob(tep_id_a+tep_id_b), self.tf.get_spe_ob(tep_id_b+tep_id_a))
            return self.tf.get_normal(tep_id_a+tep_id_b), self.tf.get_normal(tep_id_b+tep_id_a), ob_a_str, ob_b_str
        else:
            print("get sep template "+ tep_id_a+' '+tep_id_b+ "error")

    def add_Template(self, modelId, templateId, normal_get, obvious_get, color_get, area_get, draw_get, grab_get):
        """
        Add template to template factory from string's
        @param:
            modelId: model ID template belong to
            templateId: template ID
            normal_get: template normal string
            obvious_get: template obvious string
            color_get: template color string
            area_get: template area string
            draw_get: template draw point string
            grab_get: template grab poing string
        """
        tmp_template = TemplateFactory.atom_template(templateId)
        a_, b_, c_, d_, e_ = TemplateFactory.DecodeTemplate(obvious_get, color_get, area_get, draw_get, grab_get)
        tmp_template.fill(normal_get, a_, b_, c_, d_, e_)
        if self.tf.has_template(templateId):
            self.tf.reset_template(tmp_template)
        else:
            self.tf.add_template(modelId, tmp_template)
        self.tf_sync_flag = False

    def del_Template(self, templateId):
        """
        Delete a template
        @param:
            templateId: template ID
        """
        if self.tf.has_template(templateId):
            self.tf.del_template(templateId)
            self.tf_sync_flag = False
        else:
            print("删除模板失败 "+templateId+ ' error: No such template')
            raise KeyError

    def add_SpeTemplate(self,id_a, id_b, normal_a, normal_b, ob_a, ob_b):
        """
        Add Specific template to template factory
        @param:
            id_a: the template ID A
            id_b: the template ID B
            normal_a: the A template normal string
            normal_b: the B template normal string
            ob_a: the A template obvious string
            ob_b: the B template obvious string
        """
        tmp_spetemplate = TemplateFactory.spe_template(id_a, id_b)
        a_, b_ = TemplateFactory.DecodeSpeTemplate(ob_a, ob_b)
        tmp_spetemplate.fill(normal_a, normal_b, a_, b_)
        if self.tf.has_spe(id_a, id_b):
            self.tf.del_sep(id_a, id_b)
            self.tf.add_sep(tmp_spetemplate)
        else:
            self.tf.add_sep(tmp_spetemplate)
        self.tf_sync_flag = False

    def del_SpeTemplate(self, spe_id_a, spe_id_b):
        """
        Delete Specific template from template factory
        @param:
            spe_id_a: Specific template ID A
            spe_id_b: Specific template ID B
        """
        if self.tf.has_spe(spe_id_a, spe_id_b):
            self.tf.del_sep(spe_id_a, spe_id_b)
            self.tf_sync_flag = False
        else:
            print("del "+spe_id_a+' '+spe_id_b+' error: No such Spe template')
            raise KeyError

    def base64ToCvMat(self, b64_img):
        image = np.fromstring(base64.b64decode(b64_img), np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)

    def cvMatToBase64(self, image):
        buf = cv2.imencode('.jpg', image)[1]
        return str(base64.b64encode(buf))[2:-1]

    def get_gray_img(self, img_get):
        """
        Get a Gray Image from BGR image
        @param:
            img_get: Input Image must a opencv BGR Image
        """
        drop_ag, get_mag = TemplateBuilder.get_quant_ang(img_get)
        return TemplateBuilder.normal_mag(get_mag)

    def get_a_sim(self, img_a, img_b,id_a, id_b):
        pass

    #lab version
    def set_range(self, range_get):
        tmp_list = self.tf.gen_smart_list(range_get)
        for i in tmp_list:
            self.lm.load_a_temp(self.tf.get_normal(i))
            self.lm.load_a_id(i)
            #新版core load shape
            self.lm.add_single_shape(i, self.tf[i].area)
        #新版core 设置 nms
        self.lm.set_nms()
    def match_a_img(self, img_tar):
        raw_res = self.lm.match_a_img(img_tar)
        print("#DEBUG RAW_RES LEN", len(raw_res))
        #print("\n#DEBUG linemod process:", time.time()-t1)
        im_ag = self.lm.cal_quant(img_tar)
        det_res = MvDector.area_color_process(self.tf, im_ag, img_tar, raw_res)
        #print("\n#DEBUG total process:", time.time()-t1)
        return det_res

    def papare_res(self, img_tar, mode = 'HSV'):
        raw_res = self.lm.match_a_img(img_tar)
        im_ag = self.lm.cal_quant(img_tar)
        import EvDector
        return EvDector.fill_res_mp(self.tf, im_ag, img_tar, raw_res, mode)

