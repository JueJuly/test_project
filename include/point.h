
/*
 * FileName : point.h
 * Author   : July
 * Version  : v1.0
 * Date     : 2017/04/19 
 * Brief    : define point struct or class head file
 * 
 * Copyright (C) ....
 */
#ifndef POINT_H_
#define POINT_H_

/*---------------
*定义点坐标的结构 
*---------------*/
typedef struct WS_POINT2D 
{
    int x;
    int y; 

}ws_Point2d; //2维 

typedef struct WS_POINT 
{
    int x;
    int y; 

}ws_Point; 
 
typedef struct WS_POINTF2D
{
    float x;
    float y;  

}ws_Pointf2d; 

typedef struct WS_POINT2F
{
	float x;
	float y;

}ws_Point2f;
 
typedef struct WS_POINT3D
{
    int x;
    int y;
    int z; 

}ws_Point3d; //3维 
 
typedef struct WS_POINTF3D
{
    float x;
    float y; 
    float z;  

}ws_Pointf3d;

typedef struct WS_POINT3F
{
	float x;
	float y;
	float z;

}ws_Point3f;

typedef struct CORNER_PT_2F
{
	ws_Point2f *Point2f;
	int n_num;

}corner_pt_2f;

typedef struct CORNER_PT
{
	ws_Point *Point;
	int n_num;

}corner_pt;



#endif