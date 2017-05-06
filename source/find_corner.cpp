/*
 * FileName : find_corner.cpp
 * Author   : July
 * Version  : v1.0
 * Date     : 2017/04/19 
 * Brief    : this is source file, which defines the functions of finding 
			  corners correction
 * 
 * Copyright (C) ....
 */
//#include "stdafx.h"
#include "find_corner.h"


/*
 *----------------------------------------------------------
 * Brief：查找角点的测试函数
 * Return: 无
 * Param：无
 * Fan in: fine_corner();
 * Fan out：main()
 * Version:
 *		v1.0	2017.4.19 create by July，the first version
 *----------------------------------------------------------
 */
int find_corner_test()
{
	cv::Mat o_src_img = imread("./test_img2/H_left1000.bmp",IMREAD_COLOR);

	cv::Point Pt_0(200,249);
	cv::Point Pt_1(268,251);
	cv::Point Pt_2(130,278);
	cv::Point Pt_3(198,295);

	cv::Point Pt_4(475,253);
	cv::Point Pt_5(536,251);
	cv::Point Pt_6(548,294);
	cv::Point Pt_7(603,279);

	std::vector<cv::Point> Pt_vec;
	
	std::stringstream ss;
	std::string pt_name;

	Pt_vec.push_back( Pt_0 );
	Pt_vec.push_back( Pt_1 );
	Pt_vec.push_back( Pt_2 );
	Pt_vec.push_back( Pt_3 );

	Pt_vec.push_back( Pt_4 );
	Pt_vec.push_back( Pt_5 );
	Pt_vec.push_back( Pt_6 );
	Pt_vec.push_back( Pt_7 );

	for( int i = 0; i < 8; i++ )
	{
		/*ss << "Pt_" << i ;
		ss >> pt_name;
		ss.clear();
		ss.str("");*/
		 
		cv::circle( o_src_img,Pt_vec[i],1,CV_RGB(255,0,0),1,8); 

	}


	//std::vector<cv::Point2f> o_corner_set;
	//bool b_subpixel_refine = true;

	////查找图像中的棋盘角点
	//find_corner(o_src_img, o_corner_set, b_subpixel_refine);

	////将找到的角点在原图像上画出来
	//for( int i=0; i<o_corner_set.size(); i++ )
	//{
	//	cv::circle(o_src_img, cv::Point((int)o_corner_set[i].x, (int)o_corner_set[i].y),4,cv::Scalar(255,128,0) );
	//}

	cv::namedWindow("Corner_img", WINDOW_AUTOSIZE);
	cv::imshow("Corner_img", o_src_img);

	cv::waitKey(0);

	cv::destroyAllWindows();

	return 0;
}

/*
 *-------------------------------------------------------
 * Brief：实现在图像中查找棋盘格角点功能
 * Return: int类型的值(没有实际意义)
 * Param：
 *		1、src_img          in		输入的原始待检测图像 	
 *		2、&corner_points   inout	保存检测到的角点的数组
 * Fan in:
 * Fan out：find_corner_test();
 * Version:
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
int find_corner(cv::Mat src_img, std::vector<cv::Point2f> &corner_points, bool subpixel_refine)
{
	if(src_img.empty()){
		std::cout << "input the src_img is empty!" << std::endl;
		return -1;
	}

	
	int n_h = src_img.rows;
	int n_w = src_img.cols;
	int n_channel = src_img.channels();
	cv::Mat o_gray_img(src_img.size(),CV_8UC1);

	if( 1 == src_img.channels() )
	{
		o_gray_img = src_img.clone();
		//src_img.copyTo(o_gray_img);
	}
	else if( 3 == src_img.channels() )
	{
		cvtColor(src_img, o_gray_img, CV_RGB2GRAY);
	}

	GaussianBlur(o_gray_img, o_gray_img, Size(3,3),0); //gauss filter

	cv::Mat o_float_img(n_h, n_w, CV_32FC1);

	o_gray_img.convertTo(o_float_img, CV_32FC1); //将灰度图像转换成double类型的
	//src_mat.convertTo(dst_mat,CV_32F);
	cv::Mat o_grad_x_img(n_h, n_w, CV_32FC1);	//x轴方向的梯度矩阵
	cv::Mat o_grad_y_img(n_h, n_w, CV_32FC1);	//y轴方向的梯度矩阵
	cv::Mat o_angle_img(n_h, n_w, CV_32FC1);	//计算sobel后梯度的方向矩阵

	//Mat abs_grad_x, abs_grad_y;  
	Mat o_weight_img(n_h, n_w, CV_32FC1); //计算sobel后的权重矩阵
	float *pd_weight_data = NULL;
	float *pd_grad_x_data = NULL;
	float *pd_grad_y_data = NULL;
	float *pd_angle_data = NULL;

	cv::Sobel( o_gray_img, o_grad_x_img, CV_32FC1, 1, 0, 3 );
	cv::Sobel( o_gray_img, o_grad_y_img, CV_32FC1, 0, 1, 3 );

	//convertScaleAbs( o_grad_x_img, abs_grad_x );
	//convertScaleAbs( o_grad_y_img, abs_grad_y );

	//得到每个像素对应的梯度权重和角度
	for(int r = 0; r < n_h; r++)
	{
		pd_weight_data = o_weight_img.ptr<float>(r);
		pd_grad_x_data = o_grad_x_img.ptr<float>(r);
		pd_grad_y_data = o_grad_y_img.ptr<float>(r);
		pd_angle_data  = o_angle_img.ptr<float>(r);

		for(int c = 0; c < n_w; c++)
		{
			pd_weight_data[c] = sqrtf( (pd_grad_x_data[c] * pd_grad_x_data[c]) + \
									   (pd_grad_y_data[c] * pd_grad_y_data[c]) );
			pd_angle_data[c] = atan2f( pd_grad_y_data[c], pd_grad_x_data[c] );

			if( pd_angle_data[c] < 0 )
			{
				pd_angle_data[c] += C_PI;
			}
			else if( pd_angle_data[c] > C_PI )
			{
				pd_angle_data[c] -= C_PI;
			}

			/*pd_angle_data[c] = pd_angle_data[c] < 0 ?  \
								(pd_angle_data[c] + C_PI) : \
								( pd_angle_data[c] > C_PI ? (pd_angle_data[c] - C_PI): pd_angle_data[c] ) ;*/
		}
	}

	// addWeighted(src1,alpha,src2,beta,gamma,dst);
	// dst = src1*alpha + src2*beta + gamma; 
	// addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, o_weight_img ); 
	cv::Mat o_norm_img;//归一化后的矩阵
	double d_min;
	double d_max;

	//得到最大值和最小值做归一化处理
	cv::minMaxLoc(o_float_img,&d_min,&d_max); //get max and min value

	cv::Mat o_corner_img( n_h,n_w,CV_32FC1,Scalar(0) ); //角点图像

	cv::normalize(o_gray_img,o_norm_img,1.0,0.0,NORM_MINMAX);//实现图像的归一化，最大值为1，最小值为0
	
	//定义方向模板数据矩阵
	/*------------------------------
	 * template_props:
	 *	0		pi/2	4
     *	pi/4	-pi/4	4
	 *	0		pi/2	8
	 *	pi/4	-pi/4	8
	 *	0		pi/2	12
	 *	pi/4	-pi/4	12
	--------------------------------*/
	cv::Mat template_props = cv::Mat(6,3,CV_32FC1);
	template_props.at<float>(0,0) = (float)0;
	template_props.at<float>(0,1) = (float)C_PI/2;
	template_props.at<float>(0,2) = (float)4;

	template_props.at<float>(1,0) = (float)C_PI/4;
	template_props.at<float>(1,1) = (float)-C_PI/4;
	template_props.at<float>(1,2) = (float)4;

	template_props.at<float>(2,0) = (float)0;
	template_props.at<float>(2,1) = (float)C_PI/2;
	template_props.at<float>(2,2) = (float)8;

	template_props.at<float>(3,0) = (float)C_PI/4;
	template_props.at<float>(3,1) = (float)-C_PI/4;
	template_props.at<float>(3,2) = (float)8;

	template_props.at<float>(4,0) = (float)0;
	template_props.at<float>(4,1) = (float)C_PI/2;
	template_props.at<float>(4,2) = (float)12;

	template_props.at<float>(5,0) = (float)C_PI/4;
	template_props.at<float>(5,1) = (float)-C_PI/4;
	template_props.at<float>(5,2) = (float)12;

	//对应两种4个方向的角点矩阵，第一种：上、下、左、右；第二种：左上、右下、右上、左下
	cv::Mat img_corner_a1(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_a2(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_b1(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_b2(Size(n_w,n_h), CV_32FC1);

	cv::Mat sum_1(Size(n_w,n_h), CV_32FC1);
	cv::Mat sum_2(Size(n_w,n_h), CV_32FC1);
	cv::Mat sum_3(Size(n_w,n_h), CV_32FC1);
	
	//4个方向的角点均值矩阵
	cv::Mat img_corner_mu(Size(n_w,n_h), CV_32FC1);

	//临时的数据矩阵
	cv::Mat img_corner_a(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_b(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_1(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_2(Size(n_w,n_h), CV_32FC1);

	//得到的相关模板矩阵
	std::vector<cv::Mat> template_vec;

	//临时存放的角点
	std::vector<cv::Point2d> temp_save_pt;

	//根据创建的模板对图像进行滤波处理
	//目的就是得到角点
	for(int r = 0; r < template_props.rows; r++)
	{
		create_template( template_props.at<float>(r,1), template_props.at<float>(r,2), template_props.at<float>(r,3), template_vec );

		//注意：template_vec中方向模板矩阵在创建函数中存放的顺序，取出时要对应
		cv::filter2D(o_float_img, img_corner_a1, o_float_img.depth(), template_vec[0] );//a1
		cv::filter2D(o_float_img, img_corner_a2, o_float_img.depth(), template_vec[1] );//s2
		cv::filter2D(o_float_img, img_corner_b1, o_float_img.depth(), template_vec[2] );//b1
		cv::filter2D(o_float_img, img_corner_b2, o_float_img.depth(), template_vec[3] );//b2

		cv::add(img_corner_a1, img_corner_a2, sum_1);
		cv::add(img_corner_b1, img_corner_b2, sum_2);
		cv::add(sum_1, sum_2, sum_3);

		img_corner_mu = sum_3 / 4;

		//img_corner_mu = (img_corner_a1 + img_corner_a2 + img_corner_b1 + img_corner_b2)/4;

		//case 1: a = white, b = black
		//cv::min( (img_corner_a1-img_corner_mu), (img_corner_a2-img_corner_mu), img_corner_a );
		//cv::min( (img_corner_mu-img_corner_b1), (img_corner_mu-img_corner_b2), img_corner_b );
		//cv::min( img_corner_a, img_corner_b, img_corner_1 );

		////case 2: a = black, b = white
		//cv::min( (img_corner_mu-img_corner_a1), (img_corner_mu-img_corner_a2), img_corner_a );
		//cv::min( (img_corner_b1-img_corner_mu), (img_corner_b2-img_corner_mu), img_corner_b );
		//cv::min( img_corner_a, img_corner_b, img_corner_2 );

		//cv::max(  o_corner_img, img_corner_1, o_corner_img );
		//cv::max(  o_corner_img, img_corner_1, o_corner_img );

	}

	//利用非极大值抑制处理筛选出角点坐标
	non_max_suppress(o_corner_img, 3, 0.025, 5, temp_save_pt);

	//根据测试结果可以选择做不做亚像素级处理
	if( subpixel_refine )
	{
		//角点坐标亚像素级处理
		//corner_coords_subpixel_refine();
	}




}

/*
 *-------------------------------------------------------
 * Brief：根据输入的参数创建模板
 * Return: 无
 * Param：
 *		1、f_angle1			in		输入的角度1 	
 *		2、f_angle2			in		输入的角度2
 *		3、f_radius			in		输入的半径
 *		4、&template_vec	inout	返回得到的模板数组
 * Fan in:
 * Fan out：find_corner();
 * Version:
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
void create_template( float f_angle1, float f_angle2, float f_radius, std::vector<cv::Mat> &template_vec )
{
	int n_w = (int)(f_radius * 2 + 1);
	int n_h = (int)(f_radius * 2 + 1);

	float f_mid_w = f_radius ;
	float f_mid_h = f_radius ;
	float f_dist;
	float f_s1;
	float f_s2;

	double d_a1_sum;
	double d_a2_sum;
	double d_b1_sum;
	double d_b2_sum;
	cv::Scalar Sum;

	std::vector<float> vec;

	//compute normals from angles
	float f_n1[2] = { -sin(f_angle1), cos(f_angle1) };
	float f_n2[2] = { -sin(f_angle2), cos(f_angle2) };

	//创建并初始化模板矩阵
	cv::Mat a1_mat( n_h, n_w, CV_32FC1, Scalar(0) ); //或者Mat::zero(n_h, n_w, CV_32FC1); 
	cv::Mat b1_mat( n_h, n_w, CV_32FC1, Scalar(0) );
	cv::Mat a2_mat( n_h, n_w, CV_32FC1, Scalar(0) );
	cv::Mat b2_mat( n_h, n_w, CV_32FC1, Scalar(0) );

	//如果数组不为空，则清空数组，
	if( !template_vec.empty() )
	{
		template_vec.clear();
	}

	for(int w = 0; w < n_w; w++)
	{
		/*if(!vec.empty())
			vec.clear();

		if(!vec.empty())
			vec.clear();*/

		for(int h = 0; h < n_h; h++)
		{
			/*vec.push_back(w-f_mid_w);
			vec.push_back(h-f_mid_h);*/

			//f_dist = compute_vec_norm(vec);
			f_dist = (w-f_mid_w) * (w-f_mid_w) + (h-f_mid_h) * (h-f_mid_h);
			f_dist = sqrtf(f_dist);

			f_s1 = (w-f_mid_w) * (f_n1[0]) + (h-f_mid_h) * (f_n1[1]);
			f_s2 = (w-f_mid_w) * (f_n2[0]) + (h-f_mid_h) * (f_n2[1]);

			if( f_s1 <= -0.1 && f_s2 <= -0.1 ) //f_s1 <= -0.1 and f_s2 <= -0.1
			{
				a1_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, C_PI/2 );
			}
			else if(f_s1 >= 0.1 && f_s2 >= 0.1) //f_s1 >= 0.1 and f_s2 >= 0.1
			{
				a2_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, C_PI/2 );
			}
			else if(f_s1 <= -0.1 && f_s2 >= 0.1) //f_s1 <= -0.1 and f_s2 >= 0.1
			{
				b1_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, C_PI/2 );
			}
			else if(f_s1 >= 0.1 && f_s2 <= -0.1) //f_s1 >= 0.1 and f_s2 <= -0.1
			{
				b2_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, C_PI/2 );
			}
		}


	}

	//求对应矩阵的所有元素和
	Sum = cv::sum(a1_mat); //计算a1矩阵
	d_a1_sum = Sum.val[0];

	Sum = cv::sum(a2_mat); //计算a1矩阵
	d_a2_sum = Sum.val[0];

	Sum = cv::sum(b1_mat); //计算a1矩阵
	d_b1_sum = Sum.val[0];

	Sum = cv::sum(b2_mat); //计算a1矩阵
	d_b2_sum = Sum.val[0];
	
	//对每个矩阵做平均
	a1_mat /= d_a1_sum;
	a2_mat /= d_a2_sum;
	b1_mat /= d_b1_sum;
	b2_mat /= d_b2_sum;

	//将计算后的矩阵放到数组中
	template_vec.push_back(a1_mat);
	template_vec.push_back(a2_mat);
	template_vec.push_back(b1_mat);
	template_vec.push_back(b2_mat);

}

/*
 *-------------------------------------------------------
 * Brief：计算正态概率密度函数
 * Return: 在x处的正态分布的概率密度值
 * Param：
 *		1、x			in		符合正态分布的随机变量 	
 *		2、mu			in		均值，缺省为0
 *		3、sigma		in		标准方差，缺省为1
 * Fan in:
 * Fan out：create_template();
 * Version:
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
float compute_normpdf(float x, float mu, float sigma )
{
	double f_value = 0;

	if( sigma <= 0 )
	{
		std::cout << "sigma less than or equal zero!" << std::endl;
		return -1.0;
	}

	f_value = (1/(sqrtf(2*C_PI)*sigma)) * expf(-((x-mu)*(x-mu))/(2*sigma*sigma));

	return (float)f_value;

}

/*
 *-------------------------------------------------------
 * Brief：计算向量的模
 * Return: 计算后向量的模值
 * Param：
 *		1、vec	 in		待计算的向量	 	
 * Fan in:
 * Fan out：create_template();
 * Version:
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
float compute_vec_norm(std::vector<float> vec)
{
	double d_value = 0; 

	if( vec.empty() )
	{
		std::cout << "input the vec is empty!" << std::endl;
		assert( !vec.empty() );
		return 0;
	}

	for(int i = 0; i < vec.size(); i++)
	{
		d_value += vec.at(i) * vec.at(i); //平方累加和
	}

	d_value = sqrtf((float)d_value);

	return (float)d_value;

}

/*
 *---------------------------------------------------------
 * Brief：计算矩阵的2阶模,本函数只针对单通道整型矩阵
 * Return： 计算后矩阵的模值
 * Param：
 *		1、o_mat	 in		待计算的矩阵	 	
 * Fan in：
 * Fan out：create_template();
 * Version：
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
float compute_matrix_norm(cv::Mat o_mat)
{
	double d_value = 0; 
	int n_h = o_mat.rows;
	int n_w = o_mat.cols;
	float *pf_data = NULL;

	if( o_mat.empty() )
	{
		std::cout << "input the o_mat is empty!" << std::endl;
		assert( !o_mat.empty() );
		return 0;
	}

	for(int r = 0; r < n_h; r++)
	{
		pf_data = (float *)o_mat.ptr<int>(r);

		for(int c = 0; c < n_w; c++)
		{
			d_value += (double)pf_data[c] * pf_data[c]; //平方累加和
		}
	}

	d_value = sqrtf((float)d_value); //将累加后的数据开平方

	return (float)d_value;
}

/*
 *---------------------------------------------------------
 * Brief：计算所有元素的和,只针对单通道float类型的矩阵
 * Return： 返回和的值
 * Param：
 *		1、o_mat	 in		待计算的矩阵	 	
 * Fan in：
 * Fan out：create_template();
 * Version：
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
double sum_matrix(cv::Mat o_mat)
{
	if( o_mat.empty() )
	{
		std::cout << "input o_mat is empty!" << std::endl;
		assert(!o_mat.empty());
		return 0;
	}

	double d_sum = 0;
	int n_w = o_mat.cols;
	int n_h = o_mat.rows;
	float *pf_data = NULL;

	for(int h = 0; h < n_h; h++)
	{
		pf_data = o_mat.ptr<float>(h);

		for(int w = 0; w < n_w; w++)
		{
			d_sum += pf_data[w];
		}

	}

	return d_sum;

}

/*
 *---------------------------------------------------------
 * Brief：对矩阵进行归一化处理，即每一个元素除以所有元素的和
 *		  Mat.at<uchar>(h,w) = Mat.at<uchar>(h,w) / sum;
 * Return： 无
 * Param：
 *		1、o_src_mat	in		输入的原始矩阵	
 *		2、o_dst_mat	inout	传出归一化后的矩阵
 * Fan in：sum_matrix();
 * Fan out：create_template();
 * Version：
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
void mean_matrix(cv::Mat o_src_mat, cv::Mat &o_dst_mat)
{
	if( o_src_mat.empty() )
	{
		std::cout << "input o_mat is empty!" << std::endl;
		assert(!o_src_mat.empty());
		return ;
	}

	if( o_dst_mat.empty() )
	{
		std::cout << "the o_dst_mat is NULL!" << std::endl;
		assert(!o_dst_mat.empty());
		return ;
	}

	double d_sum;
	int n_h = o_src_mat.rows;
	int n_w = o_src_mat.cols;
	float *pf_src_data = NULL;
	float *pf_dst_data = NULL;

	//d_sum = sum_matrix(o_src_mat);
	cv::Scalar Sum = cv::sum(o_src_mat); //计算出每个通道的像素之和
	d_sum = Sum.val[0];


	o_dst_mat = o_src_mat/d_sum;
	/*
	for(int h = 0; h < n_h; h++)
	{
		pf_src_data = o_src_mat.ptr<float>(h);
		pf_dst_data = o_dst_mat.ptr<float>(h);

		for(int w = 0; w < n_w; w++)
		{
			pf_dst_data[h] = pf_src_data[w]/d_sum;
		}

	}*/


}

/*
 *---------------------------------------------------------
 * Brief：对输入的矩阵数据进行非极大值抑制处理，最后筛选并
 *		  输出所有极大值所处的坐标集。此函数只处理单通道的
 * Return： 无
 * Param：
 *		1、src_mat			in		输入的原始数据矩阵	
 *		2、n_region_len		in		局部搜索区域的边长
 *		3、d_threshold		in		筛选局部最大值时的阈值
 *		4、n_margin			in		最大搜索区域距图像边界的值
 *		5、&coords_points	inout	返回筛选出所有极大值的坐标
 * Fan in：
 * Fan out：find_corner();
 * Version：
 *		v1.0	2017.4.19 create by July，the first version
 *---------------------------------------------------------
 */
void non_max_suppress(cv::Mat src_mat, int n_region_len, double d_threshold, int n_margin, std::vector<cv::Point2d> &coords_points)
{
	int n_L = n_region_len;
	cv::Point2d temp_pt;
	int n_h = src_mat.rows;
	int n_w = src_mat.cols;
	float f_gray_val;
	float f_max_val;
	bool b_failed;

	//极大值的坐标
	int n_max_x;
	int n_max_y;

	if(src_mat.empty())
	{
		std::cout << "input src_mat is empty!" << std::endl;
		assert(!src_mat.empty());
		return ;
	}

	if(!coords_points.empty())
	{
		coords_points.clear();
	}

	for( int r = n_L+n_margin; r < (n_h-n_L-n_margin-1); r += (n_L+1) )
	{
		for( int c = n_L+n_margin; c < (n_w-n_L-n_margin-1); c += (n_L+1) )
		{
			temp_pt.x = c;
			temp_pt.y = r;

			f_max_val = src_mat.at<float>(r,c);

			//找局部最大值
			for( int r2 = r; r2 < (r+n_L+1); r2++ )
			{
				for( int c2 = c; c2 < (c+n_L+1); c2++ )
				{
					f_gray_val = src_mat.at<float>(r2,c2);

					if( f_gray_val > f_max_val )
					{
						n_max_x = c2;
						n_max_y = r2;

						f_max_val = f_gray_val;
					}
				}
			}

			//
			b_failed = false;
			for( int r3 = (n_max_y-n_L); r3 < min( (n_max_y+n_L+1),(n_h-n_margin) ); r3++ )
			{
				for( int c3 = (n_max_x-n_L); c3 < min( (n_max_x+n_L+1),(n_w-n_margin) ); c3++ )
				{
					f_gray_val = src_mat.at<float>(r3,c3);

					if( f_gray_val > f_max_val && ( r3 < r || r3 > (r+n_L) || c3 < c || c3 > (c + n_L) )  )
					{
						b_failed = true;
						break; //如果检测到不是极大值跳出内层循环，检测失败
					}


				}

				if(b_failed)
				{
					break; //检测失败跳出外层循环
				}


			}

			if( f_max_val >= d_threshold && !b_failed )
			{
				coords_points.push_back(temp_pt);
			}

		}
	}

	return ;

}

/*
 *---------------------------------------------------------
 * Brief：对角点坐标进行亚像素级处理
 * Return： 无
 * Param：
 *		1、grad_x_img				in		x轴方向的梯度图像	
 *		2、grad_y_img				in		y轴方向的梯度图像
 *		3、angle_img				in		梯度方向图像
 *		4、weight_img				in		梯度权重图像
 *		5、coords_points			in	    待处理的像素级角点坐标
 *		6、&corner_subpixel_coords	inout   处理后的亚像素级角点坐标
 *		7、r						in		局部矩形区域边长
 * Fan in：
 * Fan out：find_corner();
 * Version：
 *		v1.0	2017.4.20 create by July
 *---------------------------------------------------------
 */
void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img, cv::Mat angle_img, cv::Mat weight_img, 
										std::vector<cv::Point2d> corner_coords, std::vector<cv::Point2f> &corner_subpixel_coords, int r )
{
	return ;
}

/*
 *---------------------------------------------------------
 * Brief：角点相关评分，根据梯度的权重和x，y方向的角度进行评分
 * Return： 无
 * Param：
 *		1、sub_srcImg				in			原图像	
 *		2、weight_img				in			梯度权重图像
 *		3、&coords_pts_v1			inout	    对角点方向统计的一个参数
 *		4、&coords_pts_v2			inout		对角点方向统计的另一个参数
 * Fan in：
 * Fan out：find_corner();
 * Version：
 *		v1.0	2017.4.20 create by July
 *---------------------------------------------------------
 */
void corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, std::vector<ws_Data_f_2d> &coords_pts_v1, \
							      std::vector<ws_Data_f_2d> &coords_pts_v2 )
{

}


/*
 *---------------------------------------------------------
 * Brief：寻找梯度方向中两个最大峰值位置，首先是将窗口内所有
 *			梯度方向映射到一个32bin的直方图里，用梯度幅值作为
 *			加权值，然后使用meanshift方法来寻找直方图的局部最大值
 * Return： 无
 * Param：
 *		1、img_angle		in			角点角度图像	
 *		2、img_weight		in			角点梯度权重图像
 *		3、&v1				inout	    第一个峰值的位置
 *		4、&v2				inout		第二个峰值的位置
 * Fan in：
 * Fan out：find_corner();
 * Version：
 *		v1.0	2017.4.27 create by July
 *---------------------------------------------------------
 */
int edge_orientations( cv::Mat img_angle, cv::Mat img_weight, cv::vector<float> &v1, cv::vector<float> &v2 )
{
	int n_bin_num = 32;
	int n_bin = 0;

	std::vector<float> vec_angle;
	std::vector<float> vec_weight;
	std::vector<float> angle_hist;
	std::vector<ws_Data_f_2d> modes;
	std::vector<ws_Data_f_3d> modes_expand;

	std::vector<float> smoothed_hist;

	if( img_angle.empty() )
	{
		assert( !img_angle.empty() );
		return -1;
	}

	if( img_weight.empty() )
	{
		assert( !img_weight.empty() );
		return -1;
	}

	if( !v1.empty() )
	{
		v1.clear();
	}

	if( !v2.empty() )
	{
		v2.clear();
	}

	//将方向和权重图像转化为数组形式
	for( int h = 0; h < img_angle.rows; h++ )
	{
		for( int w = 0; w < img_angle.cols; w++ )
		{
			vec_angle.push_back( img_angle.at<float>(h,w) );
			vec_weight.push_back( img_weight.at<float>(h,w) );
		}
	}
	//对创建的角点直方图进行初始化
	for( int n = 0; n < n_bin_num; n++ )
	{
		angle_hist.push_back( 0 );
	}
	//将
	for( int i = 0; i < vec_angle.size(); i++ )
	{
		vec_angle[i] += C_PI/2;
		if( vec_angle[i] > C_PI )
		{
			vec_angle[i] -= C_PI;
		}
	}

	for( int i = 0; i < vec_angle.size(); i++ )
	{
		n_bin = Max( Min( vec_angle[i] / (C_PI/n_bin_num), n_bin_num-1 ), 0);
		angle_hist[n_bin] += vec_weight[i];
	}

	find_modes_meanshift( angle_hist, 1, smoothed_hist, modes );

	if( modes.size() <= 1 )
	{
		return -2;
	}

	for( int i = 0; i < modes.size(); i++ )
	{

	}


	return 0;
}

/*
 *---------------------------------------------------------
 * Brief：利用meanshift寻找局部最大值
 * Return： 无
 * Param：
 *		1、angle_hist			in			直方图	
 *		2、sigma				in			sigma值
 *		3、&hist_smoothed		inout	    平滑后的直方图
 *		4、&modes				inout		返回局部的最大值位置
 * Fan in： hist_smooth()
 * Fan out：edge_orientations();
 * Version：
 *		v1.0	2017.4.27 create by July
 *---------------------------------------------------------
 */
void find_modes_meanshift( std::vector<float> angle_hist, float sigma, std::vector<float> &hist_smoothed, \
						      std::vector<ws_Data_f_2d> &modes )
{
	//计算
	int n_times = 0;
	std::vector<float> mode_col_idx;
	std::vector<float> mode_col_val;
	ws_Data_f_2d data_f_2d;
	int n_idx;

	if( angle_hist.empty() )
	{
		assert( !angle_hist.empty() );
		return ;
	}

	//对直方图进行平滑处理
	hist_smooth( angle_hist, hist_smoothed, sigma );

	if( !modes.empty() )
	{
		modes.clear();
	}

	//检测看看是不是每个值都相等，如果都相等，后边计算出来的mode很可能会无限大
	for( n_idx = 0; n_idx < hist_smoothed.size(); n_idx++ )
	{
		if( hist_smoothed[n_idx] - hist_smoothed[0] < 1e-5 )
		{
			n_times++;
		}
	}

	
	if( hist_smoothed.size()-1 == n_times )
	{
		return ;
	}

	mode_find( hist_smoothed, mode_col_idx, mode_col_val );

	sort_mode( mode_col_idx, mode_col_val );

	for( int i = 0; i < mode_col_val.size(); i++ )
	{
		data_f_2d.x = mode_col_val[i];
		data_f_2d.y = mode_col_idx[i];

		modes.push_back( data_f_2d );
	}

}

//
/*
 *---------------------------------------------------------
 * Brief：直方图平滑处理
 * Return： 无
 * Param：
 *		1、src_hist			in			直方图	
 *		2、&dst_hist		inout	    平滑后的直方图
 *		3、sigma			in			sigma值，作用类似于高斯函数中的sigma值
 * Fan in：
 * Fan out：find_modes_meanshift();
 * Version：
 *		v1.0	2017.4.27 create by July
 *---------------------------------------------------------
 */
void hist_smooth( std::vector<float> src_hist, std::vector<float> &dst_hist, const float f_sigma )
{
	int n_start_idx = (int)(-2 * f_sigma - 0.5 );
	int n_end_idx = (int)( 2 * f_sigma + 0.5 );
	int n_idx;
	float f_sum = 0;
	int n_temp = 0;

	for( int i = 0; i < src_hist.size(); i++ )
	{
		f_sum = 0;

		for( int j = n_start_idx; j <= n_end_idx; j++ )
		{
			n_temp = i + j ;
			if( n_temp >= 0 )
			{
				n_idx =  n_temp % src_hist.size() ;
			}
			else
			{
				n_temp += src_hist.size();
				n_idx =  n_temp % src_hist.size() ;
			}
			
			f_sum += src_hist.at(n_idx) * compute_normpdf( (float)j, 0, 1.0 );
		}

		dst_hist.push_back( f_sum );
	}
}

/*
 *---------------------------------------------------------
 * Brief：mode查找函数
 * Return： 无
 * Param：
 *		1、smooth_hist			in			平滑后的直方图	
 *		2、&mode_col_1			inout	    计算出来mode的第一列数据
 *		3、&mode_col_2			inout		计算出来mode的第二列数据
 * Fan in：
 * Fan out：find_modes_meanshift();
 * Version：
 *		v1.0	2017.4.28 create by July
 *---------------------------------------------------------
 */
int mode_find( std::vector<float> smooth_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 )
{
	int n_len = smooth_hist.size();

	int n_i = 0; //外层循环的索引
	int n_j = 0; //
	int n_j1 = 0;
	int n_j2 = 0;
	float f_h0 = 0;
	float f_h1 = 0;
	float f_h2 = 0;
	int n_idx = 0;

	bool b_flag = false;

	int n_temp = 0;

	if( smooth_hist.empty() )
	{
		assert( !smooth_hist.empty() );
		return -1;
	}

	if( !mode_col_1.empty() )
	{
		mode_col_1.clear();
	}

	if( !mode_col_2.empty() )
	{
		mode_col_2.clear();
	}

	for( n_i = 0; n_i < n_len; n_i++ )
	{
		n_j = n_i;

		while( true )
		{
			f_h0 = smooth_hist.at( n_j );
			n_temp = n_j + 1;
			n_j1 = n_temp % n_len;
			n_temp = n_j - 1;

			if( n_temp < 0 )
			{
				n_temp += n_len;
				n_j2 = n_temp % n_len;
			}
			else
			{
				n_j2 = n_temp % n_len;
			}

			f_h1 = smooth_hist.at( n_j1 );
			f_h2 = smooth_hist.at( n_j2 );

			if( f_h1 >= f_h0 && f_h1 >= f_h2/*(f_h1-f_h0) >= 1e-5 && (f_h1-f_h2) >= 1e-5*/ )
			{
				n_j = n_j1;
			}
			else if( f_h2 > f_h0 && f_h2 > f_h1/*(f_h2-f_h0) > 1e-5 && (f_h2-f_h1) > 1e-5*/ )
			{
				n_j = n_j2;
			}
			else
			{
				break;
			}

			/*printf( "f_h0 = %0.4f, n_j1 = %d, n_j2 = %d, f_h1 = %0.4f, f_h2 = %0.4f, n_j = %d\n", \
				    f_h0, n_j1, n_j2, f_h1, f_h2, n_j );*/

		}

		//检测mode第一列中有与n_j相等的就跳出循环，并标记为真
		b_flag = false;

		for( n_idx = 0; n_idx < mode_col_1.size(); n_idx++ )
		{
			if( mode_col_1.at(n_idx) == n_j )
			{
				b_flag = true;
				break;
			}
		}

		if( mode_col_1.empty() || !b_flag ) //第一列为空
		{
			mode_col_1.push_back( n_j );
			mode_col_2.push_back( smooth_hist.at( n_j ) );
		}


	}

	return 0;

}

/*
 *---------------------------------------------------------
 * Brief：对mode的相关数据进行排序
 * Return： 无
 * Param：	
 *		1、&mode_col_1			inout	    排序后mode的第一列数据
 *		2、&mode_col_2			inout		排序后mode的第二列数据
 * Fan in：
 * Fan out：find_modes_meanshift();
 * Version：
 *		v1.0	2017.4.28 create by July
 *---------------------------------------------------------
 */
int sort_mode( std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 )
{
	float f_idx = 0;
	float f_val = 0;

	int i = 0,j = 0;

	if ( mode_col_1.empty() )
	{
		assert( !mode_col_1.empty() );
		return -1;
	}

	if( mode_col_2.empty() )
	{
		assert( !mode_col_2.empty() );
		return -1;
	}

	if( mode_col_1.size() != mode_col_2.size() )
	{
		assert( mode_col_1.size() == mode_col_2.size() );
		return -2;
	}

	//按照降序排列，即从大到小
	for( i = 0; i < mode_col_2.size() - 1; i++ )
	{
		for( j = 0; j < mode_col_2.size()-i-1; j++ )
		{
			if( mode_col_2[j] < mode_col_2[j+1] ) 
			{
				//值进行交换
				f_val = mode_col_2[j];
				mode_col_2[j] = mode_col_2[j+1];
				mode_col_2[j+1] = f_val;

				//索引进行交换
				f_idx = mode_col_1[j];
				mode_col_1[j] = mode_col_1[j+1];
				mode_col_1[j+1] = f_idx;
			}
		}
	}

	return 0;

}

//int sort_mode_test(  std::vector<float> src_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 )
//{
//	float f_idx = 0;
//	float f_val = 0;
//
//	int i = 0,j = 0;
//
//	if ( mode_col_1.empty() )
//	{
//		assert( !mode_col_1.empty() );
//		return -1;
//	}
//
//	if( mode_col_2.empty() )
//	{
//		assert( !mode_col_2.empty() );
//		return -1;
//	}
//
//	if( mode_col_1.size() != mode_col_2.size() )
//	{
//		assert( mode_col_1.size() == mode_col_2.size() );
//		return -2;
//	}
//
//	//按照降序排列，即从大到小
//	for( i = 0; i < mode_col_2.size() - 1; i++ )
//	{
//		for( j = 0; j < mode_col_2.size()-i-1; j++ )
//		{
//			if( mode_col_2[j] < mode_col_2[j+1] ) 
//			{
//				//值进行交换
//				f_val = mode_col_2[j];
//				mode_col_2[j] = mode_col_2[j+1];
//				mode_col_2[j+1] = f_val;
//
//				//值进行交换
//				f_idx = mode_col_1[j];
//				mode_col_1[j] = mode_col_1[j+1];
//				mode_col_1[j+1] = f_idx;
//			}
//		}
//	}
//
//	return 0;
//
//}

 int sort_test( int vec[14], int n )
 {
	int n_temp = 0;
	for( int i = 0; i < n-1; i++ )
	{
		for( int j = 0; j < n-i-1; j++ )
		{
			if( vec[j] < vec[j+1] )
			{
				n_temp = vec[j];
				vec[j] = vec[j+1];
				vec[j+1] = n_temp;
			}
		}
	}


	return 0;

 }
