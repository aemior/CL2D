#include "line2Dup.h"
#include "NmsUtility.hpp"
#include <memory>
#include <iostream>
#include <map>
#include <fstream>
#include <assert.h>
#include <opencv2/opencv.hpp>

using namespace cv;

//===============================
//  Global variables
//===============================

std::shared_ptr<line2Dup::Detector> pGDetector;
std::shared_ptr<std::vector<std::string>> pGIds;
std::shared_ptr<ShapeFactory> pGshape;
std::shared_ptr<NmsMachine> pGNms;



//===============================
//  Ctypes API
//===============================
extern "C"
{
  //计算图像梯度方向图
  uchar* img_quant(int height, int width, uchar* data_b, uchar* data_g, uchar* data_r);

  //初始化检测器唯一实例
  void initDector()
  {
    pGDetector = std::shared_ptr<line2Dup::Detector>(new line2Dup::Detector(128, {4,8}));
    pGIds = std::shared_ptr<std::vector<std::string>>(new std::vector<std::string>);
    pGshape = std::shared_ptr<ShapeFactory>(new ShapeFactory());
    pGNms = std::shared_ptr<NmsMachine>(new NmsMachine());
  }

  //向检测器内加载一个模板
  void Dector_loadTep(char* get_temp)
  {
    std::vector<std::string> v_instance;
    v_instance.push_back(get_temp);
    pGDetector->readClasses(v_instance);
  }

  //向检测器内添加一个id
  void Dector_setId(char* get_id)
  {
    pGIds->push_back(get_id);
  }

  //删除所有待匹配的id
  void Dector_cleanId()
  {
    pGIds->clear();
  }

  //nms过滤器设置
  void SetNms(){
    pGNms->set(pGshape, pGDetector, pGIds, 10);
  }

  //多边形形状操作
  bool add_a_shape(char* id_get, int16_t *shape_get, int16_t shape_len){
      std::string s;
      s = id_get;
      pGshape->add_shape(s, shape_get, shape_len);
      return true;
  }
  bool print_a_shape(char* id_get, int16_t angle_get){
      std::vector<int16_t>* print_poly;
      std::string s;
      s = id_get;
      print_poly = pGshape->get_shape(s, angle_get);
      std::cout << "#DEBUG SHAPE: ";
      for(int i=0; i< (*print_poly).size();i++){
          std::cout << (*print_poly)[i] << ' ';
      }
      std::cout << "END\n";
      return true;
  }
  void clean_shape(){
      pGshape->clean_shape();
  }

  //检测器匹配一张图片
  char* match_target(int height, int width, uchar* data_b, uchar* data_g, uchar* data_r)
  {
    //merge all data
	  cv::Mat src_b(height, width, CV_8UC1, data_b);
	  cv::Mat src_g(height, width, CV_8UC1, data_g);
	  cv::Mat src_r(height, width, CV_8UC1, data_r);
    std::vector<Mat> mbgr(3);
    mbgr[0] = src_b;
    mbgr[1] = src_g;
    mbgr[2] = src_r;
	  cv::Mat read_img;
    merge(mbgr, read_img);
    //====================
#ifdef THR
    auto matches = pGDetector->match(read_img, THR, *pGIds);
#else
    auto matches = pGDetector->match(read_img, 60, *pGIds);
#endif
    if(matches.size()==0)
    {
      std::string re_str = "NULL";
	    char* buffer = (char*)malloc(sizeof(uchar)*re_str.size()+1);
	    memcpy(buffer, re_str.c_str(), re_str.size()+1);
      return buffer;
    }
    else
    {
#ifdef MODE_NMS
      std::stringstream ss = pGNms->PolygonNms(matches);
#else
      std::stringstream ss;
      for(size_t i=0; i<matches.size(); i++)
      {
        auto match = matches[i];
        auto templ = pGDetector->getTemplates(match.class_id, match.template_id);

        float r_x = match.x + templ[0].tl_x;
        float r_y = match.y + templ[0].tl_y;

        ss << match.class_id << ' ' << match.similarity << ' ' << r_x << ' ' << r_y << ' ' << match.template_id << std::endl;
      }
#endif
      std::string re_str = ss.str();
	    char* buffer = (char*)malloc(sizeof(uchar)*re_str.size()+1);
	    memcpy(buffer, re_str.c_str(), re_str.size()+1);
      return buffer;
    }
  }
}

//===============================
//  ALL FUN
//===============================

void directGradient(Mat &magnitude, Mat &quantized_angle,Mat &angle)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;
        }
    }
    quantized_angle = quantized_unfiltered;
}

uchar* img_quant(int height, int width, uchar* data_b, uchar* data_g, uchar* data_r)
{
	cv::Mat src_b(height, width, CV_8UC1, data_b);
	cv::Mat src_g(height, width, CV_8UC1, data_g);
	cv::Mat src_r(height, width, CV_8UC1, data_r);
  std::vector<Mat> mbgr(3);
  mbgr[0] = src_b;
  mbgr[1] = src_g;
  mbgr[2] = src_r;
	cv::Mat read_img;
  merge(mbgr, read_img);
  Mat smoothed;
  Mat angle;
  Mat magnitude;
  // Compute horizontal and vertical image derivatives on all color channels separately
  static const int KERNEL_SIZE = 7;
  // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
  GaussianBlur(read_img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
  magnitude.create(read_img.size(), CV_32F);

#ifdef MODE_GRAY
  Mat gray;
  cvtColor(read_img, gray, cv::COLOR_BGR2GRAY);
  GaussianBlur(gray, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
  Mat sobel_dx, sobel_dy, sobel_ag;
  Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
  magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
#else
  // Allocate temporary buffers
  Size size = read_img.size();
  Mat sobel_3dx;              // per-channel horizontal derivative
  Mat sobel_3dy;              // per-channel vertical derivative
  Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
  Mat sobel_dy(size, CV_32F); // maximum vertical derivative
  Mat sobel_ag;               // final gradient orientation (unquantized)

  Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
  Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

  short *ptrx = (short *)sobel_3dx.data;
  short *ptry = (short *)sobel_3dy.data;
  float *ptr0x = (float *)sobel_dx.data;
  float *ptr0y = (float *)sobel_dy.data;
  float *ptrmg = (float *)magnitude.data;

  const int length1 = static_cast<const int>(sobel_3dx.step1());
  const int length2 = static_cast<const int>(sobel_3dy.step1());
  const int length3 = static_cast<const int>(sobel_dx.step1());
  const int length4 = static_cast<const int>(sobel_dy.step1());
  const int length5 = static_cast<const int>(magnitude.step1());
  const int length0 = sobel_3dy.cols * 3;

  for (int r = 0; r < sobel_3dy.rows; ++r)
  {
    int ind = 0;
    for (int i = 0; i < length0; i += 3)
    {
      // Use the gradient orientation of the channel whose magnitude is largest
      int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
      int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
      int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];
      if (mag1 >= mag2 && mag1 >= mag3)
      {
        ptr0x[ind] = ptrx[i];
        ptr0y[ind] = ptry[i];
        ptrmg[ind] = (float)mag1;
      }
      else if (mag2 >= mag1 && mag2 >= mag3)
      {
        ptr0x[ind] = ptrx[i + 1];
        ptr0y[ind] = ptry[i + 1];
        ptrmg[ind] = (float)mag2;
      }
      else
      {
        ptr0x[ind] = ptrx[i + 2];
        ptr0y[ind] = ptry[i + 2];
        ptrmg[ind] = (float)mag3;
      }
      ++ind;
    }
    ptrx += length1;
    ptry += length2;
    ptr0x += length3;
    ptr0y += length4;
    ptrmg += length5;
  }
#endif

  // Calculate the final gradient orientations
  phase(sobel_dx, sobel_dy, sobel_ag, true);
  directGradient(magnitude, angle, sobel_ag);
  //=========================
	uchar* buffer = (uchar*)malloc(sizeof(uchar)*height*width);
	memcpy(buffer, angle.data, height*width);
	return buffer;
}
