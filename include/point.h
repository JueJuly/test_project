
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
 
typedef struct WS_POINTF2D
{
    float x;
    float y;    
}ws_Pointf2d; 
 
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



#endif