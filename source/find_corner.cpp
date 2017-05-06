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
 * Brief�����ҽǵ�Ĳ��Ժ���
 * Return: ��
 * Param����
 * Fan in: fine_corner();
 * Fan out��main()
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
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

	////����ͼ���е����̽ǵ�
	//find_corner(o_src_img, o_corner_set, b_subpixel_refine);

	////���ҵ��Ľǵ���ԭͼ���ϻ�����
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
 * Brief��ʵ����ͼ���в������̸�ǵ㹦��
 * Return: int���͵�ֵ(û��ʵ������)
 * Param��
 *		1��src_img          in		�����ԭʼ�����ͼ�� 	
 *		2��&corner_points   inout	�����⵽�Ľǵ������
 * Fan in:
 * Fan out��find_corner_test();
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
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

	o_gray_img.convertTo(o_float_img, CV_32FC1); //���Ҷ�ͼ��ת����double���͵�
	//src_mat.convertTo(dst_mat,CV_32F);
	cv::Mat o_grad_x_img(n_h, n_w, CV_32FC1);	//x�᷽����ݶȾ���
	cv::Mat o_grad_y_img(n_h, n_w, CV_32FC1);	//y�᷽����ݶȾ���
	cv::Mat o_angle_img(n_h, n_w, CV_32FC1);	//����sobel���ݶȵķ������

	//Mat abs_grad_x, abs_grad_y;  
	Mat o_weight_img(n_h, n_w, CV_32FC1); //����sobel���Ȩ�ؾ���
	float *pd_weight_data = NULL;
	float *pd_grad_x_data = NULL;
	float *pd_grad_y_data = NULL;
	float *pd_angle_data = NULL;

	cv::Sobel( o_gray_img, o_grad_x_img, CV_32FC1, 1, 0, 3 );
	cv::Sobel( o_gray_img, o_grad_y_img, CV_32FC1, 0, 1, 3 );

	//convertScaleAbs( o_grad_x_img, abs_grad_x );
	//convertScaleAbs( o_grad_y_img, abs_grad_y );

	//�õ�ÿ�����ض�Ӧ���ݶ�Ȩ�غͽǶ�
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
	cv::Mat o_norm_img;//��һ����ľ���
	double d_min;
	double d_max;

	//�õ����ֵ����Сֵ����һ������
	cv::minMaxLoc(o_float_img,&d_min,&d_max); //get max and min value

	cv::Mat o_corner_img( n_h,n_w,CV_32FC1,Scalar(0) ); //�ǵ�ͼ��

	cv::normalize(o_gray_img,o_norm_img,1.0,0.0,NORM_MINMAX);//ʵ��ͼ��Ĺ�һ�������ֵΪ1����СֵΪ0
	
	//���巽��ģ�����ݾ���
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

	//��Ӧ����4������Ľǵ���󣬵�һ�֣��ϡ��¡����ң��ڶ��֣����ϡ����¡����ϡ�����
	cv::Mat img_corner_a1(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_a2(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_b1(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_b2(Size(n_w,n_h), CV_32FC1);

	cv::Mat sum_1(Size(n_w,n_h), CV_32FC1);
	cv::Mat sum_2(Size(n_w,n_h), CV_32FC1);
	cv::Mat sum_3(Size(n_w,n_h), CV_32FC1);
	
	//4������Ľǵ��ֵ����
	cv::Mat img_corner_mu(Size(n_w,n_h), CV_32FC1);

	//��ʱ�����ݾ���
	cv::Mat img_corner_a(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_b(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_1(Size(n_w,n_h), CV_32FC1);
	cv::Mat img_corner_2(Size(n_w,n_h), CV_32FC1);

	//�õ������ģ�����
	std::vector<cv::Mat> template_vec;

	//��ʱ��ŵĽǵ�
	std::vector<cv::Point2d> temp_save_pt;

	//���ݴ�����ģ���ͼ������˲�����
	//Ŀ�ľ��ǵõ��ǵ�
	for(int r = 0; r < template_props.rows; r++)
	{
		create_template( template_props.at<float>(r,1), template_props.at<float>(r,2), template_props.at<float>(r,3), template_vec );

		//ע�⣺template_vec�з���ģ������ڴ��������д�ŵ�˳��ȡ��ʱҪ��Ӧ
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

	//���÷Ǽ���ֵ���ƴ���ɸѡ���ǵ�����
	non_max_suppress(o_corner_img, 3, 0.025, 5, temp_save_pt);

	//���ݲ��Խ������ѡ�������������ؼ�����
	if( subpixel_refine )
	{
		//�ǵ����������ؼ�����
		//corner_coords_subpixel_refine();
	}




}

/*
 *-------------------------------------------------------
 * Brief����������Ĳ�������ģ��
 * Return: ��
 * Param��
 *		1��f_angle1			in		����ĽǶ�1 	
 *		2��f_angle2			in		����ĽǶ�2
 *		3��f_radius			in		����İ뾶
 *		4��&template_vec	inout	���صõ���ģ������
 * Fan in:
 * Fan out��find_corner();
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
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

	//��������ʼ��ģ�����
	cv::Mat a1_mat( n_h, n_w, CV_32FC1, Scalar(0) ); //����Mat::zero(n_h, n_w, CV_32FC1); 
	cv::Mat b1_mat( n_h, n_w, CV_32FC1, Scalar(0) );
	cv::Mat a2_mat( n_h, n_w, CV_32FC1, Scalar(0) );
	cv::Mat b2_mat( n_h, n_w, CV_32FC1, Scalar(0) );

	//������鲻Ϊ�գ���������飬
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

	//���Ӧ���������Ԫ�غ�
	Sum = cv::sum(a1_mat); //����a1����
	d_a1_sum = Sum.val[0];

	Sum = cv::sum(a2_mat); //����a1����
	d_a2_sum = Sum.val[0];

	Sum = cv::sum(b1_mat); //����a1����
	d_b1_sum = Sum.val[0];

	Sum = cv::sum(b2_mat); //����a1����
	d_b2_sum = Sum.val[0];
	
	//��ÿ��������ƽ��
	a1_mat /= d_a1_sum;
	a2_mat /= d_a2_sum;
	b1_mat /= d_b1_sum;
	b2_mat /= d_b2_sum;

	//�������ľ���ŵ�������
	template_vec.push_back(a1_mat);
	template_vec.push_back(a2_mat);
	template_vec.push_back(b1_mat);
	template_vec.push_back(b2_mat);

}

/*
 *-------------------------------------------------------
 * Brief��������̬�����ܶȺ���
 * Return: ��x������̬�ֲ��ĸ����ܶ�ֵ
 * Param��
 *		1��x			in		������̬�ֲ���������� 	
 *		2��mu			in		��ֵ��ȱʡΪ0
 *		3��sigma		in		��׼���ȱʡΪ1
 * Fan in:
 * Fan out��create_template();
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
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
 * Brief������������ģ
 * Return: �����������ģֵ
 * Param��
 *		1��vec	 in		�����������	 	
 * Fan in:
 * Fan out��create_template();
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
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
		d_value += vec.at(i) * vec.at(i); //ƽ���ۼӺ�
	}

	d_value = sqrtf((float)d_value);

	return (float)d_value;

}

/*
 *---------------------------------------------------------
 * Brief����������2��ģ,������ֻ��Ե�ͨ�����;���
 * Return�� ���������ģֵ
 * Param��
 *		1��o_mat	 in		������ľ���	 	
 * Fan in��
 * Fan out��create_template();
 * Version��
 *		v1.0	2017.4.19 create by July��the first version
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
			d_value += (double)pf_data[c] * pf_data[c]; //ƽ���ۼӺ�
		}
	}

	d_value = sqrtf((float)d_value); //���ۼӺ�����ݿ�ƽ��

	return (float)d_value;
}

/*
 *---------------------------------------------------------
 * Brief����������Ԫ�صĺ�,ֻ��Ե�ͨ��float���͵ľ���
 * Return�� ���غ͵�ֵ
 * Param��
 *		1��o_mat	 in		������ľ���	 	
 * Fan in��
 * Fan out��create_template();
 * Version��
 *		v1.0	2017.4.19 create by July��the first version
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
 * Brief���Ծ�����й�һ��������ÿһ��Ԫ�س�������Ԫ�صĺ�
 *		  Mat.at<uchar>(h,w) = Mat.at<uchar>(h,w) / sum;
 * Return�� ��
 * Param��
 *		1��o_src_mat	in		�����ԭʼ����	
 *		2��o_dst_mat	inout	������һ����ľ���
 * Fan in��sum_matrix();
 * Fan out��create_template();
 * Version��
 *		v1.0	2017.4.19 create by July��the first version
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
	cv::Scalar Sum = cv::sum(o_src_mat); //�����ÿ��ͨ��������֮��
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
 * Brief��������ľ������ݽ��зǼ���ֵ���ƴ������ɸѡ��
 *		  ������м���ֵ���������꼯���˺���ֻ����ͨ����
 * Return�� ��
 * Param��
 *		1��src_mat			in		�����ԭʼ���ݾ���	
 *		2��n_region_len		in		�ֲ���������ı߳�
 *		3��d_threshold		in		ɸѡ�ֲ����ֵʱ����ֵ
 *		4��n_margin			in		������������ͼ��߽��ֵ
 *		5��&coords_points	inout	����ɸѡ�����м���ֵ������
 * Fan in��
 * Fan out��find_corner();
 * Version��
 *		v1.0	2017.4.19 create by July��the first version
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

	//����ֵ������
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

			//�Ҿֲ����ֵ
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
						break; //�����⵽���Ǽ���ֵ�����ڲ�ѭ�������ʧ��
					}


				}

				if(b_failed)
				{
					break; //���ʧ���������ѭ��
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
 * Brief���Խǵ�������������ؼ�����
 * Return�� ��
 * Param��
 *		1��grad_x_img				in		x�᷽����ݶ�ͼ��	
 *		2��grad_y_img				in		y�᷽����ݶ�ͼ��
 *		3��angle_img				in		�ݶȷ���ͼ��
 *		4��weight_img				in		�ݶ�Ȩ��ͼ��
 *		5��coords_points			in	    ����������ؼ��ǵ�����
 *		6��&corner_subpixel_coords	inout   �����������ؼ��ǵ�����
 *		7��r						in		�ֲ���������߳�
 * Fan in��
 * Fan out��find_corner();
 * Version��
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
 * Brief���ǵ�������֣������ݶȵ�Ȩ�غ�x��y����ĽǶȽ�������
 * Return�� ��
 * Param��
 *		1��sub_srcImg				in			ԭͼ��	
 *		2��weight_img				in			�ݶ�Ȩ��ͼ��
 *		3��&coords_pts_v1			inout	    �Խǵ㷽��ͳ�Ƶ�һ������
 *		4��&coords_pts_v2			inout		�Խǵ㷽��ͳ�Ƶ���һ������
 * Fan in��
 * Fan out��find_corner();
 * Version��
 *		v1.0	2017.4.20 create by July
 *---------------------------------------------------------
 */
void corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img, std::vector<ws_Data_f_2d> &coords_pts_v1, \
							      std::vector<ws_Data_f_2d> &coords_pts_v2 )
{

}


/*
 *---------------------------------------------------------
 * Brief��Ѱ���ݶȷ�������������ֵλ�ã������ǽ�����������
 *			�ݶȷ���ӳ�䵽һ��32bin��ֱ��ͼ����ݶȷ�ֵ��Ϊ
 *			��Ȩֵ��Ȼ��ʹ��meanshift������Ѱ��ֱ��ͼ�ľֲ����ֵ
 * Return�� ��
 * Param��
 *		1��img_angle		in			�ǵ�Ƕ�ͼ��	
 *		2��img_weight		in			�ǵ��ݶ�Ȩ��ͼ��
 *		3��&v1				inout	    ��һ����ֵ��λ��
 *		4��&v2				inout		�ڶ�����ֵ��λ��
 * Fan in��
 * Fan out��find_corner();
 * Version��
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

	//�������Ȩ��ͼ��ת��Ϊ������ʽ
	for( int h = 0; h < img_angle.rows; h++ )
	{
		for( int w = 0; w < img_angle.cols; w++ )
		{
			vec_angle.push_back( img_angle.at<float>(h,w) );
			vec_weight.push_back( img_weight.at<float>(h,w) );
		}
	}
	//�Դ����Ľǵ�ֱ��ͼ���г�ʼ��
	for( int n = 0; n < n_bin_num; n++ )
	{
		angle_hist.push_back( 0 );
	}
	//��
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
 * Brief������meanshiftѰ�Ҿֲ����ֵ
 * Return�� ��
 * Param��
 *		1��angle_hist			in			ֱ��ͼ	
 *		2��sigma				in			sigmaֵ
 *		3��&hist_smoothed		inout	    ƽ�����ֱ��ͼ
 *		4��&modes				inout		���ؾֲ������ֵλ��
 * Fan in�� hist_smooth()
 * Fan out��edge_orientations();
 * Version��
 *		v1.0	2017.4.27 create by July
 *---------------------------------------------------------
 */
void find_modes_meanshift( std::vector<float> angle_hist, float sigma, std::vector<float> &hist_smoothed, \
						      std::vector<ws_Data_f_2d> &modes )
{
	//����
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

	//��ֱ��ͼ����ƽ������
	hist_smooth( angle_hist, hist_smoothed, sigma );

	if( !modes.empty() )
	{
		modes.clear();
	}

	//��⿴���ǲ���ÿ��ֵ����ȣ��������ȣ���߼��������mode�ܿ��ܻ����޴�
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
 * Brief��ֱ��ͼƽ������
 * Return�� ��
 * Param��
 *		1��src_hist			in			ֱ��ͼ	
 *		2��&dst_hist		inout	    ƽ�����ֱ��ͼ
 *		3��sigma			in			sigmaֵ�����������ڸ�˹�����е�sigmaֵ
 * Fan in��
 * Fan out��find_modes_meanshift();
 * Version��
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
 * Brief��mode���Һ���
 * Return�� ��
 * Param��
 *		1��smooth_hist			in			ƽ�����ֱ��ͼ	
 *		2��&mode_col_1			inout	    �������mode�ĵ�һ������
 *		3��&mode_col_2			inout		�������mode�ĵڶ�������
 * Fan in��
 * Fan out��find_modes_meanshift();
 * Version��
 *		v1.0	2017.4.28 create by July
 *---------------------------------------------------------
 */
int mode_find( std::vector<float> smooth_hist, std::vector<float> &mode_col_1, std::vector<float> &mode_col_2 )
{
	int n_len = smooth_hist.size();

	int n_i = 0; //���ѭ��������
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

		//���mode��һ��������n_j��ȵľ�����ѭ���������Ϊ��
		b_flag = false;

		for( n_idx = 0; n_idx < mode_col_1.size(); n_idx++ )
		{
			if( mode_col_1.at(n_idx) == n_j )
			{
				b_flag = true;
				break;
			}
		}

		if( mode_col_1.empty() || !b_flag ) //��һ��Ϊ��
		{
			mode_col_1.push_back( n_j );
			mode_col_2.push_back( smooth_hist.at( n_j ) );
		}


	}

	return 0;

}

/*
 *---------------------------------------------------------
 * Brief����mode��������ݽ�������
 * Return�� ��
 * Param��	
 *		1��&mode_col_1			inout	    �����mode�ĵ�һ������
 *		2��&mode_col_2			inout		�����mode�ĵڶ�������
 * Fan in��
 * Fan out��find_modes_meanshift();
 * Version��
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

	//���ս������У����Ӵ�С
	for( i = 0; i < mode_col_2.size() - 1; i++ )
	{
		for( j = 0; j < mode_col_2.size()-i-1; j++ )
		{
			if( mode_col_2[j] < mode_col_2[j+1] ) 
			{
				//ֵ���н���
				f_val = mode_col_2[j];
				mode_col_2[j] = mode_col_2[j+1];
				mode_col_2[j+1] = f_val;

				//�������н���
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
//	//���ս������У����Ӵ�С
//	for( i = 0; i < mode_col_2.size() - 1; i++ )
//	{
//		for( j = 0; j < mode_col_2.size()-i-1; j++ )
//		{
//			if( mode_col_2[j] < mode_col_2[j+1] ) 
//			{
//				//ֵ���н���
//				f_val = mode_col_2[j];
//				mode_col_2[j] = mode_col_2[j+1];
//				mode_col_2[j+1] = f_val;
//
//				//ֵ���н���
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
