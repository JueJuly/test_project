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
//��src_pt��Ϊ���ĵ㣬��������ƫ��offset�����������ڲ����ݶ����ĵ㣬
//���ҵ��ĵ㷵�ص�dst_pt������
int find_max_grad_corner(cv::Mat o_gray_img, cv::Point src_pt, cv::Point &dst_pt, const int offset = 5 );
//���ҽǵ㺯��
int find_corner(cv::Mat src_img, std::vector<cv::Point2f> &corner_points, bool subpixel_refine = true);

//create correlation template
void create_template( float f_angle1, float f_angle2, float f_radius, std::vector<cv::Mat> &template_vec );

//������̫�����ܶȺ���
//Normal probability density function (pdf).
float compute_normpdf(float x, float mu = 0, float sigma = 1.0);

//����������2��ģ
float compute_vec_norm(std::vector<float> vec);

//��������2��ģ
float compute_matrix_norm(cv::Mat o_mat);

//�����������Ԫ�صĺ�
double sum_matrix(cv::Mat o_mat);

//�Ծ�����й�һ��������ÿһ��Ԫ�س�������Ԫ�صĺ�Mat.at<uchar>(h,w) = Mat.at<uchar>(h,w) / sum;
void mean_matrix(cv::Mat o_src_mat, cv::Mat &o_dst_mat);

//�Ǽ���ֵ�����㷨
void non_max_suppress(cv::Mat src_mat, int n_region_len, double d_threshold, int n_margin, std::vector<cv::Point2d> &coords_points);

//�ǵ����������ؼ�������
void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img, cv::Mat angle_img, cv::Mat weight_img, 
								   std::vector<cv::Point2i> corner_coords, std::vector<cv::Point2f> &corner_subpixel_coords,
								   std::vector<cv::Point2f> &corner_v1,std::vector<cv::Point2f> &corner_v2,
								   int r );

//Ѱ���ݶȷ�������������ֵλ��
int edge_orientations( cv::Mat img_angle, cv::Mat img_weight, cv::vector<float> &v1, cv::vector<float> &v2 );

//����meanshiftѰ�Ҿֲ����ֵ
void find_modes_meanshift( std::vector<float> angle_hist, float sigma, std::vector<float> &hist_smoothed, \
						      std::vector<ws_Data_f_2d> &modes );

//ֱ��ͼƽ������
void hist_smooth( std::vector<float> src_hist, std::vector<float> &dst_hist, const float f_sigma = 1.0f );

//mode finding
int mode_find( std::vector<float> smooth_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 );

//sort mode
int sort_mode( std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 );

int sort_mode_test( std::vector<float> src_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 );

int sort_test( int vec[14], int n );

bool Eigen_Jacbi(double * pMatrix,int nDim, double *pdblVects, double *pdbEigenValues, double dbEps,int nJt)  ;
int Jacobi(double matrix[][2], double vec[][2], int maxt, int n)  ;

//�Խǵ������������
void score_corner( cv::Mat src_Mat, cv::Mat angle_img, cv::Mat weight_img, std::vector<cv::Point2d> &coords_points, 
				     std::vector<cv::Mat> &template_vec, std::vector<float> &score_corner_table );

//�ǵ�������֣������ݶȵ�Ȩ�غ�x��y����ĽǶȽ�������
float corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, ws_Data_f_2d &coords_pts_v1, \
							       ws_Data_f_2d &coords_pts_v2 );

double compute_array_std( float *src_data, int n_size, int flag = 0 );
double compute_matrix_std( cv::Mat src_mat, int flag = 0 );

double compute_array_mean( float *src_data, int n_size );
double compute_array_sum( float *src_data, int n_size );

//�������ģ��
void create_correlation_patch( float f_angle_1, float f_angle_2, std::vector<cv::Mat> &template_vec );





#endif