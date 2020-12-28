# -*- coding: utf-8 -*-  

import copy
import json

class atom_template(object):
    """
    base template object
    need a template id to init
    """
    def __init__(self, template_id):
        self.id = template_id
        self.is_empty = True
        
    def fill(self, normal, obvious, color, area, draw, grab):
        self.normal = normal
        self.obvious = obvious
        self.color = color
        self.area = area
        self.draw = draw
        if len(self.draw) == 0:
            self.have_draw = False
        else:
            self.have_draw = True
        self.grab =grab
        if len(self.grab) == 0:
            self.have_grab = False
        else:
            self.have_grab = True
        self.is_empty = False

class spe_template(object):
    """
    base Specific Template
    need id a and id b to init
    """
    def __init__(self, id_a, id_b):
        self.id_a = id_a
        self.id_b = id_b
        self.is_empty = True
    def fill(self, normal_a, normal_b, obvious_a, obvious_b):
        self.normal_a = normal_a
        self.normal_b = normal_b
        self.obvious_a = obvious_a
        self.obvious_b = obvious_b
        self.is_empty = False
    def have(self, index):
        if index == self.id_a:
            return True
        elif index == self.id_b:
            return True
        else:
            return False
    def another(self, index):
        """
        input a id return another id
        """
        if index == self.id_a:
            return self.id_b
        elif index == self.id_b:
            return self.id_a
        else:
            raise KeyError
    def ob(self, index):
        """
        use index of linmod template  to find a obvious template
        """
        if index == self.id_a + self.id_b:
            return self.obvious_a
        elif index == self.id_b + self.id_a:
            return self.obvious_b
        else:
            raise KeyError
    def nm(self, index):
        """
        use index of linmod template  to find a normal template
        """
        if index == self.id_a + self.id_b:
            return self.normal_a
        elif index == self.id_b + self.id_a:
            return self.normal_b
        else:
            raise KeyError
    def info(self, index):
        if index == self.id_a + self.id_b:
            return self.id_a
        elif index == self.id_b + self.id_b:
            return self.id_b
        else:
            raise KeyError
    def __getitem__(self, index):
        if index == self.id_a:
            return [self.normal_a, self.obvious_a]
        elif index == self.id_b:
            return [self.normal_b, self.obvious_b]
        else:
            raise KeyError

class atom_model(object):
    """
    base Model object
    use a model id to init
    """
    def __init__(self, model_id):
        self.id = model_id
        self.root_dic = {}
        self.is_empty = True
    def add_template(self, template_get):
        self.root_dic[template_get.id] = template_get
        self.is_empty = False
    def del_template(self, tmp_id_get):
        if tmp_id_get in self.root_dic.keys():
            del self.root_dic[tmp_id_get]
            if len(self.root_dic) == 0:
                self.is_empty = True
        else:
            raise KeyError
    def __getitem__(self, index):
        return self.root_dic[index]


class TemplateFactory(object):
    """
    base TemplateFactory object
    """
    def __init__(self):
        self.is_empty = True
        self.root_dic = {}
        self.template_dic ={}
        self.model_num = 0
        self.spe_dic = {}
        self.spe_idx_dic ={}
        self.spe_mirror_dic ={}

    #===========Core Function===============
    def add_model(self, mod_get):
        """
        add a model to factory
        """
        self.root_dic[mod_get.id] = mod_get
        for i in mod_get.root_dic.keys():
            self.template_dic[i] = mod_get.id
        self.model_num += 1

    def del_model(self, mod_id_get):
        """
        del a model in factory
        use model id to index
        """
        if mod_id_get in self.root_dic.keys():
            tmp_tmp_list = copy.deepcopy(self.root_dic[mod_id_get].root_dic.keys())
            for i in tmp_tmp_list:
                self.del_template(i)
            del self.root_dic[mod_id_get]
            self.model_num -= 1
        else:
            raise KeyError

    def add_sep(self, spe_tmp):
        """
        add a Specific template to factory
        """
        search_id = spe_tmp.id_a + spe_tmp.id_b
        if spe_tmp.id_a in self.spe_idx_dic.keys():
            self.spe_idx_dic[spe_tmp.id_a].append(search_id)
        else:
            self.spe_idx_dic[spe_tmp.id_a] = [search_id]
        if spe_tmp.id_b in self.spe_idx_dic.keys():
            self.spe_idx_dic[spe_tmp.id_b].append(search_id)
        else:
            self.spe_idx_dic[spe_tmp.id_b] = [search_id]
        self.spe_dic[search_id] = spe_tmp
        self.spe_mirror_dic[spe_tmp.id_b + spe_tmp.id_a] = search_id
        self.is_empty = False

    def del_sep(self, id_a_get, id_b_get):
        """
        del a Specific template in factory
        use template_id_a and template_id_b to index
        """
        search_id_a = id_a_get + id_b_get
        search_id_b = id_b_get + id_a_get
        if search_id_a in self.spe_dic.keys():
            self.spe_idx_dic[id_a_get].remove(search_id_a)
            self.spe_idx_dic[id_b_get].remove(search_id_a)
            del self.spe_dic[search_id_a]
            del self.spe_mirror_dic[search_id_b]
        elif search_id_b in self.spe_dic.keys():
            self.spe_idx_dic[id_a_get].remove(search_id_b)
            self.spe_idx_dic[id_b_get].remove(search_id_b)
            del self.spe_dic[search_id_b]
            del self.spe_mirror_dic[search_id_a]
        else:
            raise KeyError

    def add_template(self, mod_id_get, tmp_get):
        """
        add a template to factory
        need a model id
        """
        if mod_id_get in self.root_dic.keys():
            self.root_dic[mod_id_get].add_template(tmp_get)
            self.template_dic[tmp_get.id] = mod_id_get
        else:
            new_mod = atom_model(mod_id_get)
            new_mod.add_template(tmp_get)
            self.add_model(new_mod)

    def del_template(self, tmp_id_get):
        """
        del a template in factory
        use template id to index
        """
        if tmp_id_get in self.template_dic.keys():
            self.root_dic[self.template_dic[tmp_id_get]].del_template(tmp_id_get)
            if tmp_id_get in self.spe_idx_dic.keys():
                tmp_search_id_list = copy.deepcopy(self.spe_idx_dic[tmp_id_get])
                for i in tmp_search_id_list:
                    other_id = copy.deepcopy(self.spe_dic[i].another())
                    del self.spe_dic[i]
                    self.spe_idx_dic[other_id].remove(i)
                del self.spe_idx_dic[tmp_id_get]
            del self.template_dic[tmp_id_get]
        else:
            raise KeyError

    #==============================================
    #=================Custom Function=============
    def has_template(self, id_get):
        if id_get in self.template_dic.keys():
            return True
        else:
            return False
    def reset_template(self, tmp_get):
        if self.has_template(tmp_get.id):
            self.root_dic[self.template_dic[tmp_get.id]].del_template(tmp_get.id)
            self.root_dic[self.template_dic[tmp_get.id]].add_template(tmp_get)
        else:
            raise KeyError
    def get_template(self,index):
        if self.has_template(index):
            return self.root_dic[self.template_dic[index]][index].normal
        else:
            raise KeyError
    def has_model(self, mod_id_get):
        if mod_id_get in self.root_dic.keys():
            return True
        else:
            return False
    def get_modid(self, id_get):
        if id_get in self.template_dic.keys():
            return self.template_dic[id_get]
        else:
            raise KeyError
    def has_spe(self, id_get_a, id_get_b):
        if (id_get_a in self.spe_idx_dic.keys()) and (id_get_b in self.spe_idx_dic.keys()):
            return True
        else:
            return False
    def get_spe_id(self, id_get_a, id_get_b):
        if self.has_spe(id_get_a, id_get_b):
            return [id_get_a+id_get_b, id_get_b+id_get_a]
        else:
            raise KeyError
    def get_spe_ob(self, index):
        if index in self.spe_dic.keys():
            return self.spe_dic[index].ob(index)
        else:
            return self.spe_dic[self.spe_mirror_dic[index]].ob(index)
    def get_tmp_ob(self, index):
        return self.root_dic[self.template_dic[index]][index].obvious
    #==========================================
    #=============Important Function==========
    def __getitem__(self, index):
        """
        use this Function to get another info from linemod index
        both template and Specific template
        """
        if self.has_template(index):
            return self.root_dic[self.template_dic[index]][index]
        elif index in self.spe_dic.keys():
            mid_id = self.spe_dic[index].info(index)
            return self.root_dic[self.template_dic[mid_id]][mid_id]
        elif index in self.spe_mirror_dic.keys():
            mid_id = self.spe_dic[self.spe_mirror_dic[index]].info(index)
            return self.root_dic[self.template_dic[mid_id]][mid_id]
        else:
            raise KeyError
    def get_normal(self, index):
        """
        use this Function to get a linmod template from linemod index
        both template and Specific template
        """
        if self.has_template(index):
            return self.root_dic[self.template_dic[index]][index].normal
        else:
            if index in self.spe_dic.keys():
                return self.spe_dic[index].nm(index)
            else:
                return self.spe_dic[self.spe_mirror_dic[index]].nm(index)
    def gen_smart_list(self, models_range):
        re_template_ID = [] 
        for i in models_range:
            tep_in_model = self.root_dic[i].root_dic.keys()
            for j in tep_in_model:
                re_template_ID.append(j)
                if j in self.spe_idx_dic.keys():
                    spe_dic = self.spe_idx_dic[j]
                    if not spe_dic == []:
                        for k in spe_dic:
                            re_template_ID.append(k)
        return re_template_ID

def DecodeTemplate(obstr, colorstr, areastr, drawstr, grabstr):
    return json.loads(obstr), json.loads(colorstr), json.loads(areastr), json.loads(drawstr), json.loads(grabstr)
def EncodeTemplate(obvious, color, area, draw, grab):
    return json.dumps(obvious), json.dumps(color), json.dumps(area), json.dumps(draw), json.dumps(grab)
def DecodeSpeTemplate(obstr_a, obstr_b):
    return json.loads(obstr_a), json.loads(obstr_b)
def EncodeSpeTemplate(ob_a, ob_b):
    return json.dumps(ob_a), json.dumps(ob_b)
