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
#include <tchar.h>

int find_corner_test();
//以src_pt点为中心点，上下左右偏移offset个像素区域内查找梯度最大的点，
//将找到的点返回到dst_pt变量中
int find_max_grad_corner(cv::Mat o_gray_img, cv::Point src_pt, cv::Point &dst_pt, const int offset = 5 );
//查找角点函数
int find_corner(cv::Mat src_img, std::vector<cv::Point2f> &corner_points, bool subpixel_refine = true);

//create correlation template
void create_template( float f_angle1, float f_angle2, float f_radius, std::vector<cv::Mat> &template_vec );

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
void non_max_suppress(cv::Mat src_mat, int n_region_len, double d_threshold, int n_margin, std::vector<cv::Point2d> &coords_points);

//角点坐标亚像素级精简处理
void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img, cv::Mat angle_img, cv::Mat weight_img, 
								   std::vector<cv::Point2i> corner_coords, std::vector<cv::Point2f> &corner_subpixel_coords,
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
void score_corner( cv::Mat src_Mat, cv::Mat angle_img, cv::Mat weight_img, std::vector<cv::Point2d> &coords_points, 
				     std::vector<cv::Mat> &template_vec, std::vector<float> &score_corner_table );

//角点相关评分，根据梯度的权重和x，y方向的角度进行评分
float corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, ws_Data_f_2d &coords_pts_v1, \
							       ws_Data_f_2d &coords_pts_v2 );

double compute_array_std( float *src_data, int n_size, int flag = 0 );
double compute_matrix_std( cv::Mat src_mat, int flag = 0 );

double compute_array_mean( float *src_data, int n_size );
double compute_array_sum( float *src_data, int n_size );

//创建相关模板
void create_correlation_patch( float f_angle_1, float f_angle_2, std::vector<cv::Mat> &template_vec );





#endif