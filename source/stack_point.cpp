//
//#include "stdafx.h"
#include "stack_point.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>

/******************************************************
function: initial the stack of the Point2d(struct)
-------------------------------------------------------
version discription:
    v1.0 2017.4.9 create the first version by zhu zhilei
*******************************************************/
bool Init_Stack_Point2d(Stack_Point2d &S)
{
    //先构造一个空栈
    S.base = (ws_Point2d *)malloc(STACK_INIT_SIZE * sizeof(ws_Point2d));
    
    if(!S.base){
        printf("Apply memory for Stack fail!\n");
        assert(S.base != NULL);
		return false;
    } 
    
    S.top = S.base;
    S.n_stactsize = STACK_INIT_SIZE;
    
    return true;
        
}

/******************************************************
function: destroy the stack of the Point2d(struct)
-------------------------------------------------------
version discription:
    v1.0 2017.4.9 create the first version by zhu zhilei
*******************************************************/
void Destroy_Stack_Point2d(Stack_Point2d &S)
{
	S.top = NULL;
	S.n_stactsize = 0;

	if(S.base){
		free(S.base);
		return;
	}

	S.base = NULL;

	return;
}

bool Clear_Stack_Point2d(Stack_Point2d &S)
{
	if(!S.base){
        printf("the stack is emptyh!\n");
        assert(S.base != NULL);
		return false;
    }

	S.top = S.base;
	S.n_stactsize = 0;

	return true;
}

bool Stack_Empty_Point2d(Stack_Point2d &S)
{
	if(!S.base){
        printf("the stack is emptyh!\n");
        assert(S.base != NULL);
		return false;
    }

	if(S.base == S.top){
		return true;
	}

	return false;
}

int Stack_Length_Point2d(Stack_Point2d S)
{
	if(!S.base){
        printf("the stack is emptyh!\n");
        assert(S.base != NULL);
		return -1;
    }

	return (S.top - S.base);
}

bool Get_Top_Point2d(Stack_Point2d S,ws_Point2d &e)
{
	if(!S.base){
        printf("the stack is emptyh!\n");
        assert(S.base != NULL);
		return false;
    }

	if( S.top == S.base){
		return false;
	}

	e = *(S.top - 1);
	return true;
}

bool Push_Stack_Point2d(Stack_Point2d &S,ws_Point2d e)
{
	if(!S.base){
        printf("the stack is emptyh!\n");
        assert(S.base != NULL);
		return false;
    }

	if(S.top - S.base >= S.n_stactsize){
		S.base = (ws_Point2d *)realloc(S.base,(S.n_stactsize + STACK_INCREMENT) * sizeof(ws_Point2d));

		if(!S.base){
			printf("Apply memory fail!\n");
			assert(S.base != NULL);
			return false;
		 }

		S.top = S.base + S.n_stactsize;
	}

	*(S.top++) = e;

	return true;

}


bool Pop_Stack_Point2d(Stack_Point2d &S,ws_Point2d &e)
{
	if(!S.base){
        printf("the stack is emptyh!\n");
        assert(S.base != NULL);
		return false;
	}

	if(S.top == S.base){
		return false;
	}

	e = *(--S.top);

	return true;
}

void Stack_Point2d_test()
{
	Stack_Point2d S1;
	ws_Point2d point;
	ws_Point2d p1;
	int n_num;

	Init_Stack_Point2d(S1);

	point.x = 4;
	point.y = 4;

	Push_Stack_Point2d(S1,point);

	point.x = 6;
	point.y = 6;

	Push_Stack_Point2d(S1,point);

	point.x = 8;
	point.y = 8;

	Push_Stack_Point2d(S1,point);

	n_num = Stack_Length_Point2d(S1);

	printf("the length of the stack_point2d is %d\n",n_num);

	Pop_Stack_Point2d(S1,p1);

	printf("point p1(%d,%d)\n",p1.x,p1.y);

	Clear_Stack_Point2d(S1);

	Destroy_Stack_Point2d(S1);

}

