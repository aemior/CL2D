#include <iostream>
#include <map>
#include <vector>
#include <memory>
#include <sstream> 
#include "wykobi.hpp"
#include "wykobi_utilities.hpp"
#include "line2Dup.h"

class ShapeFactory
{
  public:
    ShapeFactory();
    bool add_shape(std::string &id_get, int16_t *shape_get, int16_t shape_len);
    bool clean_shape();
    std::vector<int16_t>* get_shape(std::string &id_get, int in_ang);
  private:
    std::map<std::string, std::vector<std::vector<int16_t>>> shape_pool;
    std::map<std::string, int16_t> shape_len_pool;
};


class NmsMachine
{
    public:
        NmsMachine();
        void set(std::shared_ptr<ShapeFactory> shapes_get, 
                   std::shared_ptr<line2Dup::Detector> det_get,
                   std::shared_ptr<std::vector<std::string>> ids_get,
                   int angle_thr_get);
        std::stringstream PolygonNms(std::vector<line2Dup::Match> &matchs_get);
        bool IsInserction(
            std::vector<line2Dup::Match> &matchs_get,
            std::vector<std::vector<std::vector<wykobi::segment<float,2>>>> &all_segs,
            std::vector<bool> &porcess_flag,
            std::vector<std::vector<float>> &c_xy,
            int i,
            int j
        );
    private:
        std::shared_ptr<ShapeFactory> shapes;
        std::shared_ptr<line2Dup::Detector> det;
        std::shared_ptr<std::vector<std::string>> ids;
        int angle_thr;
        std::map<std::string, int> segs_dict;
};


