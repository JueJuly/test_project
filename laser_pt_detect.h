
#ifndef LASER_PT_DETECT_H
#define LASER_PT_DETECT_H

	#ifdef WIN32
		#ifdef DLL_EXPORTS
		#define EXPORT_CLASS   __declspec(dllexport)
		#define EXPORT_API  extern "C" __declspec(dllexport)
		#else
		#define EXPORT_CLASS   __declspec(dllimport )
		#define EXPORT_API  extern "C" __declspec(dllimport )
		#endif
	#else
		#define EXPORT_CLASS
		#define EXPORT_API
	#endif

//C++
#include <ios>
#include <ctime>
#include <string>
#include <vector>
#include <iosfwd>
#include <cassert>
#include <iostream>
//OpenCV
//#include <opencv/cvaux.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

#pragma once

#pragma warning(once:4244)
#pragma warning(once:4309)
#pragma warning(once:4996)

#ifndef PI
#define PI 3.1415926 //圆周率
#endif

#ifndef Max
#define Max(a,b,c) ( (MAX(a,b)) > (c) ? (MAX(a,b)) : (c) )
#endif

#ifndef Min
#define Min(a,b,c) ( (MIN(a,b)) < (c) ? (MIN(a,b)) : (c) )
#endif

const int n_norm_height = 524;    //对校正后的roi区域图像进行规范化的高度
const int n_norm_width = 960;     //对校正后的roi区域图像进行规范化的宽度

//激光的颜色类型
enum laser_type{
	RED_LASER = 0,
	GREEN_LASER = 1,
};
/*-------------------------------------------------
 * 激光点定位类
 * 
 * 包含激光点的检测、激光点坐在区域的检测和校正
 *-------------------------------------------------
 */
class EXPORT_CLASS LaserPointPos 
{

public:
	LaserPointPos();
	~LaserPointPos();

	int m_laser_detect_execute( void *p_data, int n_width, int n_height, float &f_x, float &f_y, laser_type type = GREEN_LASER );
	int m_laser_detect_execute( cv::Mat o_input_img, float &f_x, float &f_y, laser_type type = GREEN_LASER );

	int m_roi_detect( Mat o_input_img, laser_type type = GREEN_LASER ); 
	int m_line_detect( Mat o_input_img );
	int m_roi_point_detect( Mat o_bin_img );
	Mat m_roi_img_adjust( Mat o_input_img );
	int m_laser_point_detect( Mat o_input_image ); 
	int m_laser_point_detect1( Mat o_input_img );
	int m_green_laser_point_detect( Mat o_input_img );
	int m_green_laser_point_detect1( Mat o_input_img );
	int m_green_laser_point_detect_by_hsv( Mat o_input_img );
	int m_rgb_to_hsv( uchar uc_r, uchar uc_g, uchar uc_b, float &f_h, float &f_s, float &f_v );
	int m_draw_laser_rect( Mat &o_input_img, Point2f o_laser_pos_pt, const cv::Scalar &o_color , int n_offset = 2, int n_thickness = 1 );
	int m_draw_rect( cv::Mat &input_img, cv::Point pt, CvScalar &color, int offset = 4, int thickness = 2 );
	IplImage *m_polygonal_fitting( IplImage *p_bin_img );
	int m_contours_filter( IplImage *p_bin_img, IplImage * img_8uc3, double d_area_thre );
	int m_fill_inter_contours( IplImage *p_bin_img, IplImage *p_color_img, double d_area_thre );

public:

	Mat om_dst_img;                    //用矩形框画出激光点的彩色图像
	Mat om_roi_img;                    //检测出来的roi区域图像，即从原图像上把roi区域抠出来
	Mat om_adjust_roi_img;            //经过校正的roi区域图像，即图像中只有roi区域，且已经做过形状校正处理
	Mat om_adjust_roi_bin_img;       //经过校正的roi区域二值图像
	Mat om_roi_bin_img;
	Mat om_norm_roi_img;
	vector<Point> om_roi_points;     //roi区域四边形的四个顶点

	CvSize2D32f o_roi_rect_size;     //roi区域最小旋转矩形的大小尺寸
	CvPoint2D32f o_roi_rect_center;  //roi区域最小旋转矩形的中心点坐标(是在原图像中的)
	CvRect o_min_bounding_rect;      //roi区域

	Point m_laser_point_coords;     //激光点在原始图像中的坐标
	Point2f m_adjust_laser_pt;        //激光点在校正后roi区域图像中的坐标的比例
	Point m_norm_img_laser_pt;      //激光点在规范化后图像中的坐标
    int count;

	
	
};


#endif