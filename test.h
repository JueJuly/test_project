/*
 * FileName : test.h
 * Author   : July
 * Version  : v1.0
 * Date     : 2017/04/19 
 * Brief    : test head file
 * 
 * Copyright (C) ....
 */
#ifndef TEST_H
#define TEST_H

#include "common.h"
#include "find_corner.h"

#define NDEBUG
#include <assert.h>

//void get_mask_img();
void get_mask_img(const cv::String src_img_path, const cv::String mask_zone_img_path, const cv::String save_mask_img_path );
void chessboard_corner_detect_test( const cv::String src_img_path, const cv::Size board_size );

void Bilater_Filter_Test(double **output_img,double **in_data,int row,int clo,int size,double spa_singa,double range_singa);

void test1();

void *fspace_1d(int col, int length);
void **fspace_2d(int row, int col, int lenth);
void ***fspace_3d(int row1,int row2,int row3,int lenth);
void ffree_1d(void *a);
void ffree_2d(void **a, int row);
void ffree_3d(void ***a,int row1,int row2);

static void ResizeImage(unsigned char* pSrc,int src_w,int src_h,
						unsigned char* pDst,int dst_w, int dst_h);

static void _ieInterpImageBilinear8UC1_Ver3_RowFilter(unsigned char* src, long* dst, int len, int* leftIdx, int* rightIdx, long* weight, int shift);

void Resize_image_2( const unsigned char *p_srcimg_data, unsigned char *p_dstimg_data, const int n_srcimg_w, const int n_srcimg_h, const int n_dstimg_w, const int n_dstimg_h );
void Resize_image_3( const unsigned char *p_srcimg_data, const int n_srcimg_widthstep, unsigned char *p_dstimg_data, const int n_dstimg_widthstep, \
					int n_srcimg_w, int n_srcimg_h, int n_dstimg_w, int n_dstimg_h, int n_channels = 1 );

void Resize_image_4( const unsigned char *p_srcimg_data, const int n_srcimg_widthstep, unsigned char *p_dstimg_data, const int n_dstimg_widthstep, \
					int n_srcimg_w, int n_srcimg_h, int n_dstimg_w, int n_dstimg_h, int n_channels = 1 );

void Resize_image_3();
void Resize_image_4();

void Image2double_test();

void writeDataToFileTest();

#endif