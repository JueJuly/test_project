/*
 * FileName : common.h
 * Author   : July
 * Version  : v1.0
 * Date     : 2017/04/19 
 * Brief    : common class or struct head file
 * 
 * Copyright (C) ....
 */
#ifndef COMMON_H_
#define COMMON_H_
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

//#include <cmath>
#include <math.h>
#include <iosfwd>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <iomanip>

using namespace std;

#ifndef Max
#define Max(a,b) ( (a) > (b) ? (a) : (b) )

#endif

#ifndef Min
#define Min(a,b) ( (a) < (b) ? (a) : (b) )

#endif

#ifndef Abs
#define Abs(a) ( (a) > 0 ? (a) : (-a) )

#endif

//const double C_PI = 3.141593;

#ifndef  C_PI 
#define  C_PI 3.141593

#endif // ! C_PI 3.141593

#ifndef EPS
#define EPS 0.00001  

#endif

/*---------------
*定义2维、3维的数据结构 
*---------------*/
typedef struct WS_DATA_2D 
{
    int x;
    int y; 
}ws_Data_2d; //2维 
 
typedef struct WS_DATA_F_2D
{
    float x;
    float y;    
}ws_Data_f_2d; 

typedef struct WS_DATA_D_2D
{
	double x;
	double y;
}ws_Data_d_2d;
 
typedef struct WS_DATA_3D
{
    int x;
    int y;
    int z; 
}ws_Data_3d; //3维 
 
typedef struct WS_DATA_F_3D
{
    float x;
    float y; 
    float z;   
}ws_Data_f_3d;

typedef struct WS_DATA_D_3D
{
	double x;
	double y;
	double z;
}ws_Data_d_3d;


#endif