/*
 * FileName : stack_point.h
 * Author   : July
 * Version  : v1.0
 * Date     : 2017/04/19 
 * Brief    : define the stack about point process head file
 * 
 * Copyright (C) ....
 */
#ifndef STACK_POINT_H_
#define STACK_POINT_H_


#include "point.h"
#define NDEBUG
#include <assert.h>

#define STACK_INIT_SIZE 100
#define STACK_INCREMENT 10   
     
typedef struct {
    ws_Point2d  *base;
    ws_Point2d  *top;
    int n_stactsize;   
}Stack_Point2d;

bool Init_Stack_Point2d(Stack_Point2d &S);
void Destroy_Stack_Point2d(Stack_Point2d &s);
bool Clear_Stack_Point2d(Stack_Point2d &s);
bool Stack_Empty_Point2d(Stack_Point2d &s);
int Stack_Length_Point2d(Stack_Point2d s);
bool Get_Top_Point2d(Stack_Point2d S,ws_Point2d &e);
bool Push_Stack_Point2d(Stack_Point2d &s,ws_Point2d e);
bool Pop_Stack_Point2d(Stack_Point2d &s,ws_Point2d &e);
//bool Stack_Traverse_Point2d();

void Stack_Point2d_test();

#endif