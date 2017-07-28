/*
 * FileName : find_corner.h
 * Author   : July
 * Version  : v1.0
 * Date     : 2017/04/19 
 * Brief    : find corner function head file
 * 
 * Copyright (C) ....
 */
#ifndef FIND_CORNER_H_
#define FIND_CORNER_H_

#include "common.h"
#include "point.h"
#include <tchar.h>

//定义算子类型
enum OperatorType
{
	SOBEL   = 0, //sobel算子
	LAPLACE = 1  //拉普拉斯算子
};




namespace FindCorner_ShiThomas
{
	typedef struct MV_POINT
	{
		int x;
		int y;
	}mvPoint;

	typedef struct  MV_FEATURE
	{
		float fval;
		mvPoint coord;
		int nNum;
	}mvFeature;

	template<typename T> struct greaterThanPtr
	{
		bool operator()(const T* a, const T* b) const { return *a > *b; }
	};

	bool CMP( const mvFeature &a, const mvFeature &b );

	int CornerDetect( cv::Mat grayImg, vector<cv::Point2f> &corner,int &ncorner, vector<mvFeature> &vecFeature, \
		int nMaxCornerNum, double dqualityLevel = 0.01, int nMinDistance = 10, int nblockSize = 3 );

	int cornerMinEigenVal( cv::Mat grayImg, cv::Mat &eig, int nblockSize = 3, int nkSize = 3);
	int cornerMinEigenVal_c( unsigned char *grayImgData, int w, int h, float *eigData, \
								int nblockSize = 3, int nkSize = 3 );

	int boxFilter( cv::Mat srcImg, cv::Mat dstImg, int nblockSize = 3);
	//int filter_c( unsigned char *srcImg, int w, int h, unsigned char *dstImg,int blockSize = 3);
	int filter_c( float *srcImg, int w, int h, float *dstImg,int blockSize = 3, bool bnormalize = true );

	int sobel_x( unsigned char *srcImg, int w, int h, float *dstImg, 
					int ksize = 3, double dscale = 1.0,double ddelta = 0.0 );

	int sobel_y( unsigned char *srcImg, int w, int h, float *dstImg, 
					int ksize = 3, double dscale = 1.0,double ddelta = 0.0 );

	int maxMinValLoc( cv::Mat img, double &dMaxVal, cv::Point &maxValPos, double &dMinVal, cv::Point &minValPos);
	int getMaxVal( float *srcData, int w, int h, double &dmaxVal );
	int threshold_ToZero( float *srcData, float *dstData, int w, int h, double threshold );
	int dilate_c( float *srcData, float *dstData, int w, int h );
	int sort_cplusplus( vector<float *> &tempCorner, bool ascend = true );
	int sort_c( mvFeature *cornerFeature, bool ascend = true );

	int run();

}

int print_data_int( unsigned char *p_data, int sx, int ex, int sy, int ey, int n_w );
int print_data_float( float *pf_data, int sx, int ex, int sy, int ey, int n_w );
int print_data_mat( cv::Mat data, int sx, int ex, int sy, int ey);

int find_corner_test();
//以src_pt点为中心点，上下左右偏移offset个像素区域内查找梯度最大的点，
//将找到的点返回到dst_pt变量中
int find_max_grad_corner(cv::Mat o_gray_img, cv::Point src_pt, cv::Point &dst_pt, 
						    const int offset = 5, const enum OperatorType operator_type = LAPLACE );

int find_max_grad_corner_test();
//查找角点函数
int find_corner(cv::Mat gray_img, std::vector<cv::Point2f> &corner_points, 
				float f_thresh = 0.01, bool is_subpixel_refine = true);

//将uchar型的三通道图像归一化后转化成单通道的float型图像数据,
void convert_img_uchar_to_float( cv::Mat src_img_uchar, cv::Mat dst_img_float );

//create correlation template
void create_template( float f_angle1, float f_angle2, float f_radius, 
					    std::vector<cv::Mat> &template_vec );

//矩阵的卷积计算，
void matrix_convolve_compute( cv::Mat src_mat, cv::Mat &dst_mat, cv::Mat kernel_mat );

//矩阵的卷积计算，图像的边缘不做处理
void matrix_convolve_compute_1( cv::Mat src_mat, cv::Mat &dst_mat, cv::Mat kernel_mat );

//两个矩阵对应位置的最小值保存到dstMat中
void min_matrix( cv::Mat srcMat_1, cv::Mat srcMat_2, cv::Mat &dstMat );
//两个矩阵对应位置的最小值保存到dstMat中
void max_matrix( cv::Mat srcMat_1, cv::Mat srcMat_2, cv::Mat &dstMat );

//计算正太概率密度函数
//Normal probability density function (pdf).
float compute_normpdf(float x, float mu = 0, float sigma = 1.0);

//计算向量的2阶模
float compute_vec_norm(std::vector<float> vec);

//计算矩阵的2阶模
float compute_matrix_norm(cv::Mat o_mat);

//计算矩阵所有元素的和
double sum_matrix(cv::Mat o_mat);

//对矩阵进行归一化处理，即每一个元素除以所有元素的和Mat.at<uchar>(h,w) = Mat.at<uchar>(h,w) / sum;
void mean_matrix(cv::Mat o_src_mat, cv::Mat &o_dst_mat);

//非极大值抑制算法
void non_max_suppress(cv::Mat src_mat, int n_region_len, double d_threshold, int n_margin, std::vector<cv::Point> &coords_points);

//角点坐标亚像素级精简处理
void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img, cv::Mat angle_img, cv::Mat weight_img, 
								   std::vector<cv::Point> corner_coords, std::vector<cv::Point2f> &corner_subpixel_coords,
								   std::vector<cv::Point2f> &corner_v1,std::vector<cv::Point2f> &corner_v2,
								   int r );

//寻找梯度方向中两个最大峰值位置
int edge_orientations( cv::Mat img_angle, cv::Mat img_weight, cv::vector<float> &v1, cv::vector<float> &v2 );

//利用meanshift寻找局部最大值
void find_modes_meanshift( std::vector<float> angle_hist, float sigma, std::vector<float> &hist_smoothed, \
						      std::vector<ws_Data_f_2d> &modes );

//直方图平滑处理
void hist_smooth( std::vector<float> src_hist, std::vector<float> &dst_hist, const float f_sigma = 1.0f );

//mode finding
int mode_find( std::vector<float> smooth_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 );

//sort mode
int sort_mode( std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 );

int sort_mode_test( std::vector<float> src_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 );

int sort_test( int vec[14], int n );

bool Eigen_Jacbi(double * pMatrix,int nDim, double *pdblVects, double *pdbEigenValues, double dbEps,int nJt)  ;
int Jacobi(double matrix[][2], double vec[][2], int maxt, int n)  ;

//对角点进行评分排序
void score_corner( cv::Mat src_Mat, cv::Mat angle_img, cv::Mat weight_img, std::vector<cv::Point2f> corner_subpixel_coords, 
					 std::vector<cv::Point2f> corner_v1,std::vector<cv::Point2f> corner_v2,
				     std::vector<int> radius, std::vector<float> &score_corner_table );

int round(float f_data);

//角点相关评分，根据梯度的权重和x，y方向的角度进行评分
float corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, ws_Data_f_2d coords_pts_v1, \
							       ws_Data_f_2d coords_pts_v2 );

double compute_array_std( float *src_data, int n_size, int flag = 0 );
double compute_matrix_std( cv::Mat src_mat, int flag = 0 );

double compute_array_mean( float *src_data, int n_size );
double compute_array_sum( float *src_data, int n_size );

//查找数组中最大值和最小值
void find_array_max_min_val( float *src_data, int n_size, float *f_max_val, float *f_min_val );

int compute_array_dot_product( float *input_array1, float *input_array2, float *output_array, int n_array_size );

//创建相关模板
void create_correlation_patch( float f_angle_1, float f_angle_2, std::vector<cv::Mat> &template_vec );

//单方向的sobel滤波
int single_direct_sobel( const float *p_src_data, float *p_dst_data, 
						    const int n_width, const int n_height );

int find_corner( unsigned char *p_gray_img, int n_height, int n_width, CornerPt2f corner_pt );



#endif