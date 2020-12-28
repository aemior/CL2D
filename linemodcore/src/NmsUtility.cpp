#include <iostream>
#include <vector>
#include <map>
#include "wykobi.hpp"
#include "wykobi_utilities.hpp"
#include "NmsUtility.hpp"

ShapeFactory::ShapeFactory(){};
NmsMachine::NmsMachine(){};

bool ShapeFactory::add_shape(std::string &id_get, int16_t *shape_get, int16_t shape_len)
{
  std::vector<std::vector<int16_t>> tmp_shape;
  for(int i=0; i<360; i++){
    std::vector<int16_t> angle_shape;
    for(int j=0; j<shape_len; j++){
      angle_shape.push_back(shape_get[i*shape_len+j]);
    }
    tmp_shape.push_back(angle_shape);
  }
  shape_pool[id_get] = tmp_shape;
  shape_len_pool[id_get] = shape_len;
  return true;
};


std::vector<int16_t>* ShapeFactory::get_shape(std::string &id_get, int in_ang){
    return &shape_pool[id_get][in_ang];
};

bool ShapeFactory::clean_shape(){
    shape_pool.clear();
    shape_len_pool.clear();
    return true;
}

void NmsMachine::set(std::shared_ptr<ShapeFactory> shapes_get, 
                   std::shared_ptr<line2Dup::Detector> det_get,
                   std::shared_ptr<std::vector<std::string>> ids_get,
                   int angle_thr_get)
{
    shapes = shapes_get;
    det = det_get;
    ids = ids_get;
    angle_thr = angle_thr_get;
    for(int i=0;i<(*ids).size();i++){
        segs_dict[(*ids)[i]] = i;
    }
}

std::stringstream NmsMachine::PolygonNms(std::vector<line2Dup::Match> &matchs_get) 
{
    /* ss 为输出字符串流
    *  has_process为表示是否生成过线段格式的flag
    *  has_del为表示剔除的Flag
    * rx_ry 存储处理过的中心点
    * segs_pool 按类别存储所有的线段格式[类别][序号][第n条线段]，其中[类别]的索引范围
    * 和ids的数量有关，[序号]的索引范围为matchs的长度，有冗余
    */
    std::stringstream ss;
    std::vector<bool> has_process(matchs_get.size(), false);
    std::vector<bool> has_del(matchs_get.size(), false);
    std::vector<float> rx_ry_tmp(2);
    std::vector<std::vector<float>> rx_ry(matchs_get.size(), rx_ry_tmp);
    std::vector<std::vector<std::vector<wykobi::segment<float,2>>>> segs_pool((*ids).size());
    std::vector<int> class_dict(matchs_get.size());
    for(int i=0; i<(*ids).size(); i++){
        int shape_len = (*(shapes->get_shape((*ids)[i],0))).size()/2;
        std::vector<wykobi::segment<float,2>> seg_tmp(shape_len);
        std::vector<std::vector<wykobi::segment<float,2>>> segs(matchs_get.size(), seg_tmp);
        segs_pool[i] = segs;
    }
    //First Loop
    {
        auto match = matchs_get[0];
        auto templ = det->getTemplates(match.class_id, match.template_id);
        float r_x = match.x + templ[0].tl_x;
        float r_y = match.y + templ[0].tl_y;
        ss << matchs_get[0].class_id << ' ' << matchs_get[0].similarity << ' ' << r_x << ' ' << r_y << ' ' << matchs_get[0].template_id << std::endl;
        int no_1 = segs_dict[matchs_get[0].class_id];
        rx_ry[0][0] = r_x;
        rx_ry[0][1] = r_y;
        for(int j = 1; j<matchs_get.size(); j++){
            templ = det->getTemplates(matchs_get[j].class_id, matchs_get[j].template_id);
            float x_2 = matchs_get[j].x + templ[0].tl_x;
            float y_2 = matchs_get[j].y + templ[0].tl_y;
            rx_ry[j][0] = x_2;
            rx_ry[j][1] = y_2;
            class_dict[j] = segs_dict[matchs_get[j].class_id];
            if(((r_x-x_2)>200)||((r_y-y_2) > 200))
                continue;
            if(no_1 == class_dict[j]){
                if(abs(matchs_get[j].template_id - match.template_id) < angle_thr){
                    bool ins = IsInserction(matchs_get,segs_pool, has_process, rx_ry, 0, j);
                    if(ins)
                        has_del[j] = true;
                }
            }
        }
    }
    //Rest Loop
    for(int i = 0; i<matchs_get.size()-1; i++){
        if(!has_del[i]){
            auto match = matchs_get[i];
            ss << matchs_get[i].class_id << ' ' << matchs_get[i].similarity << ' ' << rx_ry[i][0] << ' ' << rx_ry[i][1] << ' ' << matchs_get[i].template_id << std::endl;
            for(int j = i+1; j<matchs_get.size(); j++){
                if(!has_del[j]){
                    if(class_dict[i] == class_dict[j]){
                        if(((rx_ry[i][0]-rx_ry[j][0])>200)||((rx_ry[i][1]-rx_ry[j][1]) > 200))
                            continue;
                        if(abs(matchs_get[i].template_id - matchs_get[j].template_id) < angle_thr){
                            bool ins = IsInserction(matchs_get,segs_pool, has_process, rx_ry, i, j);
                            if(ins)
                                has_del[j] = true;
                        }
                    }
                    
                }
            }
        }
    }
   return ss; 
};

bool NmsMachine::IsInserction(
            std::vector<line2Dup::Match> &matchs_get,
            std::vector<std::vector<std::vector<wykobi::segment<float,2>>>> &all_segs,
            std::vector<bool> &porcess_flag,
            std::vector<std::vector<float>> &c_xy,
            int i,
            int j)
{
    /*
    *判断两个多边形是否相交
    *初次处理的话会把线段格式的多边形存储到all_segs里面
    */
    int idx = segs_dict[matchs_get[j].class_id];
    int seg_size = all_segs[idx][i].size(); 
    if(!porcess_flag[i]){
        std::vector<int16_t>* shape_t = shapes->get_shape(matchs_get[i].class_id, matchs_get[i].template_id);
        for(int m=0;m<(seg_size); m++){
            all_segs[idx][i][m] = wykobi::make_segment((*shape_t)[m*2]+c_xy[i][0], (*shape_t)[m*2+1]+c_xy[i][1], (*shape_t)[(m+1)*2]+c_xy[i][0], (*shape_t)[(m+1)*2+1]+c_xy[i][1]);
        }
        porcess_flag[i] = true;
    }
    if(!porcess_flag[j]){
        std::vector<int16_t>* shape_t = shapes->get_shape(matchs_get[j].class_id, matchs_get[j].template_id);
        std::vector<wykobi::segment<float,2>> tmp_segs((*shape_t).size());
        for(int m=0;m<(seg_size); m++){
            all_segs[idx][j][m] = wykobi::make_segment((*shape_t)[m*2]+c_xy[j][0], (*shape_t)[m*2+1]+c_xy[j][1], (*shape_t)[(m+1)*2]+c_xy[j][0], (*shape_t)[(m+1)*2+1]+c_xy[j][1]);
        }
        porcess_flag[j] = true;
    }
    for(int m=0;m<seg_size;m++){
        for(int n=0;n<seg_size;n++){
            if(wykobi::intersect(all_segs[idx][i][m],all_segs[idx][j][n]))
                return true;
        }
    }
    return false;
}