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
	cv::Mat o_src_img = imread("./ChangAn_2/img24.bmp",IMREAD_COLOR);
	//cv::Mat o_clone_img = o_src_img.clone();

	cv::Mat o_gray_img;
	cvtColor( o_src_img, o_gray_img, CV_RGB2GRAY );
	
	std::vector<cv::Point2f> o_corner_set;
	bool b_subpixel_refine = true;
	float f_thresh = 0.01;

	std::stringstream ss;
	std::string str_text;

	//����ͼ���е����̽ǵ�
	find_corner( o_src_img, o_corner_set, f_thresh, b_subpixel_refine );

	//���ҵ��Ľǵ���ԭͼ���ϻ�����
	for( int i=0; i<o_corner_set.size(); i++ )
	{
		cv::circle(o_src_img, cv::Point((int)o_corner_set[i].x, (int)o_corner_set[i].y),4,cv::Scalar(255,128,0) );
		ss << i;
		ss >> str_text;
		ss.clear();
		ss.str("");
		putText(o_src_img,str_text,Point((int)o_corner_set[i].x-5, (int)o_corner_set[i].y-5),
				CV_FONT_HERSHEY_PLAIN, 1.2, CV_RGB(0,255,0),2);
	}

	cv::namedWindow("Corner_img", WINDOW_AUTOSIZE);
	cv::imshow("Corner_img", o_src_img);

	cv::waitKey(0);

	cv::destroyAllWindows();

	return 0;
}

/*
 *----------------------------------------------------------
 * Brief���������ǵ�Ĳ��Ժ���
 * Return: ��
 * Param����
 * Fan in: main()
 * Fan out��find_max_grad_corner()
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
 *----------------------------------------------------------
 */
int find_max_grad_corner_test()
{
	cv::Mat o_src_img = imread("./ChangAn_2/Front.bmp",IMREAD_COLOR);
	cv::Mat o_clone_img = o_src_img.clone();

	cv::Mat o_gray_img;
	cvtColor( o_src_img, o_gray_img, CV_RGB2GRAY );
	GaussianBlur( o_gray_img, o_gray_img, cv::Size(3,3), 0, 0 );

	//����
	cv::Point Pt_0(202,245);//(200,249)
	cv::Point Pt_1(263,251);//(268,251)
	cv::Point Pt_2(135,273);//(130,278)
	cv::Point Pt_3(198,290);//(198,295)

	cv::Point Pt_4(471,250);//(475,253)
	cv::Point Pt_5(532,255);//(536,251)
	cv::Point Pt_6(544,290);//(548,294)
	cv::Point Pt_7(605,275);//(603,279)

	std::vector<cv::Point> Pt_vec;
	cv::Point dst_pt;
	
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

	//Laplacian( src_gray, dst, CV_16S, 1, 1, 0, BORDER_DEFAULT ); 

	for( int i = 0; i < 8; i++ )
	{
		find_max_grad_corner( o_gray_img, Pt_vec[i], dst_pt, 7, LAPLACE );
		cv::circle( o_clone_img,dst_pt,1,CV_RGB(0,255,0),1,8); 
		printf("dst_pt%d = %d,%d\n",i,dst_pt.x,dst_pt.y);
	}

	cv::namedWindow("Corner_img", WINDOW_AUTOSIZE);
	cv::imshow("Corner_img", o_src_img);

	cv::namedWindow("Clone_img", WINDOW_AUTOSIZE);
	cv::imshow("Clone_img", o_clone_img);

	cv::waitKey(0);

	cv::destroyAllWindows();

	return 0;
}
 /*
 *----------------------------------------------------------
 * Brief�������ݶ����Ľǵ㣬��src_pt��Ϊ���ĵ㣬
		  ��������ƫ��offset�����������ڲ����ݶ����ĵ㣬
//���ҵ��ĵ㷵�ص�dst_pt������
 * Return: int��������
 * Param��
 *		1��cv::Mat					o_gray_img		in		�����ԭʼ�Ҷ�ͼ��
 *		2��cv::Point				src_pt			in		ԭʼ�ǵ�����
 *		3��cv::Point				&dst_pt			inout	�ݶ����Ľǵ�����
 *		4��const int				offset			in		ƫ������Ĭ��Ϊ5
 *		5��const enum OperatorType  operator_type	in		�����ݶ�ʱʹ�õ����ӣ�Ĭ����laplace
 * Fan in: 
 * Fan out����
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
 *----------------------------------------------------------
 */
int find_max_grad_corner( cv::Mat o_gray_img, cv::Point src_pt, cv::Point &dst_pt, 
							const int offset,const enum OperatorType  operator_type )
{
	int h = o_gray_img.rows;
	int w = o_gray_img.cols;
	double grad_xy = 0;
	//std::vector<float> grad_vec;
	double max_grad = 0;

	double sub_x =0;
	double sub_y =0;

	if( o_gray_img.empty() )
	{
		assert( !o_gray_img.empty() );
		std::cout << "the gray image is empty!" << endl;
		return -1;
	}

	max_grad = 0;

	//sobel����
	if( SOBEL == operator_type )
	{
		for( int r = src_pt.y-offset; r < src_pt.y+offset; r++ )
		{
			for( int c = src_pt.x-offset; c < src_pt.x+offset; c++ )
			{
				sub_x = std::abs( o_gray_img.at<uchar>(r-1,c+1) - o_gray_img.at<uchar>(r-1,c-1) + 
					(o_gray_img.at<uchar>(r,c+1) - o_gray_img.at<uchar>(r,c-1) ) * 2 +
					o_gray_img.at<uchar>(r+1,c+1) - o_gray_img.at<uchar>(r+1,c-1) );

				sub_y = std::abs( o_gray_img.at<uchar>(r+1,c-1) - o_gray_img.at<uchar>(r-1,c-1) + 
					(o_gray_img.at<uchar>(r+1,c) - o_gray_img.at<uchar>(r-1,c) )* 2 +
					o_gray_img.at<uchar>(r+1,c+1) - o_gray_img.at<uchar>(r-1,c+1) );

				grad_xy = (float)sqrt( (double)( sub_x * sub_x + sub_y * sub_y ) );

				if( grad_xy - max_grad > 1e-3 )
				{
					max_grad = grad_xy;
					dst_pt.x = c;
					dst_pt.y = r;

				}

			}//end for c
		}//end for r
	}//end if SOBEL

	//Laplace����
	if( LAPLACE == operator_type )
	{
		for( int r = src_pt.y-offset; r < src_pt.y+offset; r++ )
		{
			for( int c = src_pt.x-offset; c < src_pt.x+offset; c++ )
			{
				sub_x = (double)std::abs( o_gray_img.at<uchar>(r-1,c-1) + 
					o_gray_img.at<uchar>(r-1,c)    + 
					o_gray_img.at<uchar>(r-1,c+1)  +
					o_gray_img.at<uchar>(r,c-1)   +
					o_gray_img.at<uchar>(r,c+1)   +
					o_gray_img.at<uchar>(r+1,c-1) +  
					o_gray_img.at<uchar>(r+1,c)   + 
					o_gray_img.at<uchar>(r+1,c+1) - 
					o_gray_img.at<uchar>(r,c) * 8);

				grad_xy = sqrt( (double)( sub_x * sub_x ) );

				if( grad_xy > max_grad )
				{
					max_grad = grad_xy;
					dst_pt.x = c;
					dst_pt.y = r;

				}

			}//end for c
		}// end for r
	}//end if LAPLACE 

	return 0;

}

 /*
 *-------------------------------------------------------
 * Brief����uchar�͵���ͨ��ͼ��ת���ɵ�ͨ����float��ͼ������
 *		  ��ͨ����float��������ʱ��һ����ģ���0����1,���Ǵ˴���
 *		  ��һ�������������ֵ����Сֵ���һ�������ǽ���Сֵ
 *		  
 * Return: ��
 * Param��
 *		1��cv::Mat src_img_uchar	in	�������1 	
 *		2��cv::Mat dst_img_double	in	�������2
 * Fan in: find_corner()
 * Fan out��
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
 *---------------------------------------------------------
 */
 void convert_img_uchar_to_float( cv::Mat src_img_uchar, cv::Mat dst_img_float )
 {
	 int n_h = 0;
	 int n_w = 0;
	 int h = 0;
	 int w = 0;
	 
	 float f_R = 0;
	 float f_G = 0;
	 float f_B = 0;

	 n_h = src_img_uchar.rows;
	 n_w = src_img_uchar.cols;

	 uchar *pu8_src_data = NULL;
	 float *pf_dst_data = NULL;

	 if( src_img_uchar.empty() || dst_img_float.empty() )
	 {
		 printf("src_img or dst_img is empty!\n");
		 return ;
	 }

	for( h = 0; h < n_h; h++ )
	{
		//�õ���i�е�ָ�룬�ȼ��� p = src_img_uchar.data + h*src_img_uchar.step
		pu8_src_data = src_img_uchar.ptr<uchar>(h);
		pf_dst_data = dst_img_float.ptr<float>(h);

		for ( w = 0; w < n_w; w++ )
		{
			f_B = (float)pu8_src_data[w*3]/255;
			f_G = (float)pu8_src_data[w*3+1]/255;
			f_R = (float)pu8_src_data[w*3+2]/255;

			pf_dst_data[w] = 0.2989 * f_R + 0.5870 * f_G + 0.1140 * f_B;

		}

	}

 }

/*
 *-------------------------------------------------------
 * Brief��ʵ����ͼ���в������̸�ǵ㹦��
 * Return: int���͵�ֵ(û��ʵ������)
 * Param��
 *		1��cv::Mat					src_img				in		�����ԭʼ�����ͼ�� 	
 *		2��std::vector<cv::Point2f> &corner_points		inout	�����⵽�Ľǵ������
 *		3��float					f_thresh			in		�������ֵ�������ֱȽϵ���ֵ��Ĭ��Ϊ0.01
 *		3��bool						is_subpixel_refine	in		�Ƿ���������ش���Ĭ��Ϊture
 * Fan in: find_corner_test()
 * Fan out��create_template(),non_max_suppress(),corner_coords_subpixel_refine()
 *			score_corner()
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
 *---------------------------------------------------------
 */
int find_corner(cv::Mat src_img, std::vector<cv::Point2f> &corner_points, float f_thresh, bool is_subpixel_refine)
{
	if( src_img.empty() )
	{
		std::cout << "input the src_img is empty!" << std::endl;
		return -1;
	}

	int n_h = src_img.rows;
	int n_w = src_img.cols;
	int n_channel = src_img.channels();

	cv::Mat o_gray_img(src_img.size(),CV_8UC1);
	cv::Mat o_float_img = Mat::zeros(n_h,n_w,CV_32FC1);
	cv::Mat o_norm_img = Mat::zeros(n_h,n_w,CV_32FC1);//��һ����ľ���
	if( 1 == src_img.channels() )
	{
		o_gray_img = src_img.clone();
		//src_img.copyTo(o_gray_img);
	}
	else if( 3 == src_img.channels() )
	{
		//src_img = src_img/255;
		cvtColor(src_img, o_gray_img, CV_RGB2GRAY);
	}

	//GaussianBlur(o_gray_img, o_gray_img, Size(3,3),0); //gauss filter

	convert_img_uchar_to_float( src_img, o_norm_img );

	printf("checking if the converted image data(float type and normalization) is correct!\n");
	printf("------------------------------------------------!\n");
	//
	/*for( int h = 0; h < 20; h++ )
	{
		for( int w = 0; w < 20; w++ )
		{
			printf("%0.4f,", o_norm_img.at<float>(h,w) );
		}
		printf("\n");
		printf("\n");
	}*/
	
	cv::Mat o_grad_x_img(n_h, n_w, CV_32FC1);	//x�᷽����ݶȾ���
	cv::Mat o_grad_y_img(n_h, n_w, CV_32FC1);	//y�᷽����ݶȾ���
	cv::Mat o_angle_img(n_h, n_w, CV_32FC1);	//����sobel���ݶȵķ������ 
	cv::Mat o_weight_img(n_h, n_w, CV_32FC1); //����sobel���Ȩ�ؾ���

	cv::Mat sub_mat_1 = Mat::zeros(n_h, n_w, CV_32FC1);
	cv::Mat sun_mat_2 = Mat::zeros(n_h, n_w, CV_32FC1);

	float *pd_weight_data = NULL;
	float *pd_grad_x_data = NULL;
	float *pd_grad_y_data = NULL;
	float *pd_angle_data = NULL;

	cv::Sobel( o_norm_img, o_grad_x_img, CV_32FC1, 1, 0, 3 );
	cv::Sobel( o_norm_img, o_grad_y_img, CV_32FC1, 0, 1, 3 );

	//Ϊ�˺�matlab�е��м�����һ�������޸�
	o_grad_x_img = o_grad_x_img * -1;
	o_grad_y_img = o_grad_y_img * -1;

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
			pd_weight_data[c] = sqrtf( (pd_grad_x_data[c] * pd_grad_x_data[c]) + (pd_grad_y_data[c] * pd_grad_y_data[c]) );
			pd_angle_data[c] = atan2f( pd_grad_y_data[c], pd_grad_x_data[c] );

			//�������ĽǶȽ�����0����PI
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

	
	double d_min = 0;
	double d_max = 0;
	std::vector<int> radius_vec;

	//�õ����ֵ����Сֵ����һ������
	cv::minMaxLoc(o_norm_img,&d_min,&d_max); //get max and min value

	o_float_img = o_norm_img * 1/(float)d_max;

	cv::Mat o_corner_img = Mat::zeros( n_h,n_w,CV_32FC1 ); //�ǵ�ͼ��

	/*for( int h = 0; h < 20; h++ )
	{
		for( int w = 0; w < 20; w++ )
		{
			printf("%0.4f,", o_angle_img.at<float>(h,w) );
		}
		printf("\n");
		printf("\n");
	}
	*/
	printf("checking if the grad_x image and grad_y and angle_img and weight_img data is correct!\n");
	printf("------------------------------------------------!\n");
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
	std::vector<cv::Point> temp_corner_coords;
	std::vector<cv::Point2f> corner_v1;
	std::vector<cv::Point2f> corner_v2;
	std::vector<cv::Point2f> temp_corner_subpixel_coords;

	std::vector<float> score_corner_vec;

	std::cout << "start filter......................." << endl;

	//���ݴ�����ģ���ͼ������˲�����
	//Ŀ�ľ��ǵõ��ǵ�
	for(int r = 0; r < template_props.rows; r++)
	{
		create_template( template_props.at<float>(r,0), template_props.at<float>(r,1), template_props.at<float>(r,2), template_vec );

		
		printf("%0.4f,%0.4f,%0.4f\n\n", template_props.at<float>(r,0), template_props.at<float>(r,1), template_props.at<float>(r,2) );
		
		//ע�⣺template_vec�з���ģ������ڴ��������д�ŵ�˳��ȡ��ʱҪ��Ӧ
		cv::filter2D(o_float_img, img_corner_a1, o_float_img.depth(), template_vec[0] );//a1
		cv::filter2D(o_float_img, img_corner_a2, o_float_img.depth(), template_vec[1] );//a2
		cv::filter2D(o_float_img, img_corner_b1, o_float_img.depth(), template_vec[2] );//b1
		cv::filter2D(o_float_img, img_corner_b2, o_float_img.depth(), template_vec[3] );//b2
		for( int h = 0; h < template_vec[0].rows; h++ )
		{
			for( int w = 0; w < template_vec[0].cols; w++ )
			{
				printf("%0.4f,", template_vec[0].at<float>(h,w) );
			}
			printf("\n");
			printf("\n");
		}
		std::cout << "�ǵ�ͼ������" << endl;
		for( int h = 0; h < 20; h++ )
		{
			for( int w = 0; w < 20; w++ )
			{
				printf("%0.4f,", o_float_img.at<float>(h,w) );
			}
			printf("\n");
			printf("\n");
		}

		cv::add(img_corner_a1, img_corner_a2, sum_1);
		cv::add(img_corner_b1, img_corner_b2, sum_2);
		cv::add(sum_1, sum_2, sum_3);

		img_corner_mu = sum_3 / 4;

		//img_corner_mu = (img_corner_a1 + img_corner_a2 + img_corner_b1 + img_corner_b2)/4;
		//sub_mat_1 = Mat::zeros(n_h, n_w, CV_32FC1);
		//sun_mat_2 = Mat::zeros(n_h, n_w, CV_32FC1);

		//sub_mat_1 = img_corner_a1-img_corner_mu;
		//sun_mat_2 = img_corner_a2-img_corner_mu;
		//case 1: a = white, b = black
		//min_matrix( (img_corner_a1-img_corner_mu), (img_corner_a2-img_corner_mu), img_corner_a );
		min_matrix( (img_corner_a1-img_corner_mu), (img_corner_a2-img_corner_mu), img_corner_a );
		min_matrix( (img_corner_mu-img_corner_b1), (img_corner_mu-img_corner_b2), img_corner_b );
		min_matrix( img_corner_a, img_corner_b, img_corner_1 );

		//case 2: a = black, b = white
		min_matrix( (img_corner_mu-img_corner_a1), (img_corner_mu-img_corner_a2), img_corner_a );
		min_matrix( (img_corner_b1-img_corner_mu), (img_corner_b2-img_corner_mu), img_corner_b );
		min_matrix( img_corner_a, img_corner_b, img_corner_2 );

		max_matrix(  o_corner_img, img_corner_1, o_corner_img );
		max_matrix(  o_corner_img, img_corner_2, o_corner_img );

	}

	
	

	//���÷Ǽ���ֵ���ƴ���ɸѡ���ǵ�����
	non_max_suppress(o_corner_img, 3, 0.025, 5, temp_corner_coords);

	std::cout << "non_max_suppress is end!" << endl;

	std::cout << "start Refining......................." << endl;

	//���ݲ��Խ������ѡ�������������ؼ�����
	if( is_subpixel_refine )
	{
		//�ǵ����������ؼ�����
		corner_coords_subpixel_refine( o_grad_x_img, o_grad_y_img, o_angle_img, o_weight_img, temp_corner_coords,	
											temp_corner_subpixel_coords, corner_v1, corner_v2, 10 );
	}

	//�Ƴ����Ǳ�Ե�Ľǵ�
	for( int i = 0; i < corner_v1.size(); i++ )
	{
		if( 0 == corner_v1[i].x && 0 == corner_v1[i].y )
		{
			temp_corner_subpixel_coords.erase( temp_corner_subpixel_coords.begin()+i );
			corner_v1.erase( corner_v1.begin()+i );
			corner_v2.erase( corner_v2.begin()+i );
		}
	}

	std::cout << "Start Scoring ............................" << endl;

	radius_vec.push_back( 4 );
	radius_vec.push_back( 8 );
	radius_vec.push_back( 12 );

	//�Խǵ��������
	score_corner( o_float_img, o_angle_img, o_weight_img, temp_corner_subpixel_coords,
					corner_v1, corner_v2, radius_vec, score_corner_vec );

	//�Ƴ����ֵ͵Ľǵ�
	for( int i = 0; i < score_corner_vec.size(); i++ )
	{
		if( score_corner_vec[i] < f_thresh )
		{
			temp_corner_subpixel_coords.erase( temp_corner_subpixel_coords.begin()+i );
			corner_v1.erase( corner_v1.begin()+i );
			corner_v2.erase( corner_v2.begin()+i );
			score_corner_vec.erase( score_corner_vec.begin()+i );
		}
	}

	//for( int i = 0; i < 
	corner_points.assign( temp_corner_subpixel_coords.begin(), temp_corner_subpixel_coords.end() );

	return 0;

}

/*
 *-------------------------------------------------------
 * Brief�� ���ú˾���kernel_mat��src_mat������о��������
 *			�õ�dst_mat����,���������dst_mat�ߴ��src_matһ��
 * Return: ��
 * Param��
 *		1��cv::Mat srcMat_1		in		����Դ����1 	
 *		2��cv::Mat dst_mat		inout	�������2
 *		3��cv::Mat kernel_mat	in		�����
 * Fan in: find_corner()
 * Fan out��
 * Version:
 *		v1.0	2017.5.18 create by July��the first version
 * Note:
 *		ֻ��Ե�ͨ����ͼ��
 *---------------------------------------------------------
 */
void matrix_convolve_compute( cv::Mat src_mat, cv::Mat dst_mat, cv::Mat kernel_mat )
{
	int n_h = src_mat.rows;
	int n_w = src_mat.cols;

	int n_r = kernel_mat.rows;
	int n_c = kernel_mat.cols;

	float *pf_src_data = NULL;
	float *pf_dst_data = NULL;
	float *pf_kernel_data = NULL;

	float f_temp_val = 0;

	for( int h = 0; h < n_h; h++ )
	{
		//pf_src_data = src_mat.ptr<float>(h);
		pf_dst_data = dst_mat.ptr<float>(h);

		for( int w = 0; w < n_w; w++ )
		{
			//����߽�
			if( h <= (int)n_r/2 )//�ϱ߽�
			{
				if( w <= (int)n_c/2 )//��߽�
				{
					f_temp_val = 0;

					for( int r = n_r-(int)(n_r/2)-h-1; r < n_r; r++ )
					{
						pf_kernel_data = kernel_mat.ptr<float>(r);
						pf_src_data = src_mat.ptr<float>(r-h);

						for( int c = n_c-(int)(n_c/2)-w-1; c < n_c; c++ )
						{
							f_temp_val += pf_kernel_data[c] * ;
						}
					}
				}
				else if( w >= n_w-(int)n_c/2-1 )//�ұ߽�
				{

				}
				
			}
			else if( h >= n_h-(int)n_r/2-1 )//�±߽�
			{
				if( w <= (int)n_c/2 )//��߽�
				{
					for( int r = n_r-(int)(n_r/2)-h-1; r < n_r; r++ )
					{
						for( int c = n_c-(int)(n_c/2)-w-1; c < n_c; c++ )
						{

						}
					}
				}
				else if( w >= n_w-(int)n_c/2-1 )//�ұ߽�
				{

				}
			}

		}//end for w
	}//end for h

}

/*
 *-------------------------------------------------------
 * Brief�����������Ӧλ�õ���Сֵ���浽dstMat��
 * Return: ��
 * Param��
 *		1��cv::Mat srcMat_1		in		�������1 	
 *		2��cv::Mat srcMat_2		in		�������2
 *		3��cv::Mat &dstMat		inout	�õ����������Ӧλ�õ���Сֵ
 * Fan in: find_corner()
 * Fan out��
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
 *---------------------------------------------------------
 */
void min_matrix( cv::Mat srcMat_1, cv::Mat srcMat_2, cv::Mat &dstMat )
{
	int n_height = srcMat_1.rows;
	int n_width  = srcMat_1.cols;

	if( srcMat_1.empty() || srcMat_2.empty() || srcMat_1.rows != srcMat_2.rows || srcMat_1.cols != srcMat_2.cols )
	{
		return;
	}

	for( int h = 0; h < n_height; h++ )
	{
		for( int w = 0; w < n_width; w++ )
		{
			dstMat.at<float>(h,w) = Min( srcMat_1.at<float>(h,w), srcMat_2.at<float>(h,w) );
		}//end for w
	}//end for h

}

/*
 *-------------------------------------------------------
 * Brief�����������Ӧλ�õ����ֵ���浽dstMat��
 * Return: ��
 * Param��
 *		1��cv::Mat srcMat_1		in		�������1 	
 *		2��cv::Mat srcMat_2		in		�������2
 *		3��cv::Mat &dstMat		inout	�õ����������Ӧλ�õ����ֵ
 * Fan in: find_corner()
 * Fan out��
 * Version:
 *		v1.0	2017.4.19 create by July��the first version
 *---------------------------------------------------------
 */
void max_matrix( cv::Mat srcMat_1, cv::Mat srcMat_2, cv::Mat &dstMat )
{
	int n_height = srcMat_1.rows;
	int n_width  = srcMat_1.cols;

	if( srcMat_1.empty() || srcMat_2.empty() || srcMat_1.rows != srcMat_2.rows || srcMat_1.cols != srcMat_2.cols )
	{
		return;
	}

	for( int h = 0; h < n_height; h++ )
	{
		for( int w = 0; w < n_width; w++ )
		{
			dstMat.at<float>(h,w) = Max( srcMat_1.at<float>(h,w), srcMat_2.at<float>(h,w) );
		}//end for w
	}//end for h
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
	//cv::Scalar Sum;

	std::vector<float> vec;

	//compute normals from angles
	float f_n1[2] = { -sin(f_angle1), cos(f_angle1) };
	float f_n2[2] = { -sin(f_angle2), cos(f_angle2) };

	//��������ʼ��ģ�����
	cv::Mat a1_mat = Mat::zeros(n_h, n_w, CV_32FC1);//( n_h, n_w, CV_32FC1, Scalar(0) ); 
	cv::Mat b1_mat = Mat::zeros(n_h, n_w, CV_32FC1);//( n_h, n_w, CV_32FC1, Scalar(0) );
	cv::Mat a2_mat = Mat::zeros(n_h, n_w, CV_32FC1);//( n_h, n_w, CV_32FC1, Scalar(0) );
	cv::Mat b2_mat = Mat::zeros(n_h, n_w, CV_32FC1);//( n_h, n_w, CV_32FC1, Scalar(0) );

	//������鲻Ϊ�գ���������飬
	if( !template_vec.empty() )
	{
		template_vec.clear();
	}

	for(int w = 0; w < n_w; w++)
	{

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
				a1_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, f_radius/2 );
			}
			else if(f_s1 >= 0.1 && f_s2 >= 0.1) //f_s1 >= 0.1 and f_s2 >= 0.1
			{
				a2_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, f_radius/2 );
			}
			else if(f_s1 <= -0.1 && f_s2 >= 0.1) //f_s1 <= -0.1 and f_s2 >= 0.1
			{
				b1_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, f_radius/2 );
			}
			else if(f_s1 >= 0.1 && f_s2 <= -0.1) //f_s1 >= 0.1 and f_s2 <= -0.1
			{
				b2_mat.at<float>(h,w) = compute_normpdf(f_dist, 0, f_radius/2 );
			}
		}


	}

	//���Ӧ���������Ԫ�غ�
	d_a1_sum = sum_matrix(a1_mat);	//����a1����
	d_a2_sum = sum_matrix(a2_mat);	//����a2����
	d_b1_sum = sum_matrix(b1_mat);	//����b1����
	d_b2_sum = sum_matrix(b2_mat);	//����b2����
	
	//��ÿ��������ƽ��
	a1_mat = a1_mat / (float)d_a1_sum;
	a2_mat = a2_mat / (float)d_a2_sum;
	b1_mat = b1_mat / (float)d_b1_sum;
	b2_mat = b2_mat / (float)d_b2_sum;

	//�������ľ���ŵ�������
	//�ڳ���Ŀ�ʼ���Ѿ���template_vec�����
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

	pf_data = NULL;

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
 * Brief����������ĺ�
 * Return�� double���͵������
 * Param��
 *		1��float *src_data	in	���������������	
 *		2��int   n_size		in	����Ĵ�С
 * Fan in��;
 * Fan out��corner_correlation_score();
 * Version��
 *		v1.0	2017.5.16 create by July��the first version
 *---------------------------------------------------------
 */
double compute_array_sum( float *src_data, int n_size )
{
	double d_ret_val = 0;
	if( NULL == src_data )
	{
		d_ret_val = -1;
		return d_ret_val;
	}

	d_ret_val = 0;

	for( int i = 0; i < n_size; i++ )
	{
		d_ret_val += *(src_data+i);
	}

	return d_ret_val;
}
/*
 *---------------------------------------------------------
 * Brief����������ı�׼�����flag��ѡ���־λ��
 *		  ���flag == 0���ڼ��㷽��ʱ����n-1֮���ٿ�����
 *		  ���flag == 1���ڼ��㷽��ʱ����n֮���ٿ�����
 *		  flagĬ��Ϊ0
 * Return�� double���͵ķ���ֵ
 * Param��
 *		1��float *src_data	in	���������������	
 *		2��int   n_size		in	����Ĵ�С
 *		2��int   flag	    in	��־ʹ�����ּ��㷽ʽ
 * Fan in��;
 * Fan out��corner_correlation_score();
 * Version��
 *		v1.0	2017.5.16 create by July��the first version
 *---------------------------------------------------------
 */
double compute_array_std( float *src_data,int n_size, int flag )
{
	double d_ret_val = 0;
	double d_sum = 0;
	double d_mean = 0;

	if( NULL == src_data )
	{
		d_ret_val  -1;
		return d_ret_val;
	}

	if( 0 == flag )
	{
		d_sum = compute_array_sum( src_data, n_size );
		d_mean = d_sum / n_size;
		d_ret_val = 0;

		for( int i = 0; i < n_size; i++ )
		{
			d_ret_val += ( *(src_data+i) - d_mean ) * ( *(src_data+i) - d_mean );
		}

		d_ret_val /= (n_size - 1);

		d_ret_val = (double)sqrtf((float)d_ret_val);

	}
	else if( 1 == flag )
	{
		d_sum = compute_array_sum( src_data, n_size );
		d_mean = d_sum / n_size;
		d_ret_val = 0;

		for( int i = 0; i < n_size; i++ )
		{
			d_ret_val += ( *(src_data+i) - d_mean ) * ( *(src_data+i) - d_mean );
		}

		d_ret_val /= n_size;

		d_ret_val = (double)sqrtf((float)d_ret_val);
	}

	return d_ret_val;
}
/*
 *---------------------------------------------------------
 * Brief���Ծ����׼����㣬flag��һ���жϱ�־
 *		  
 * Return�� ��
 * Param��
 *		1��cv::Mat  src_mat	in		�����ԭʼ����	
 *		2��int		flag    inout	������һ����ľ���
 * Fan in��sum_matrix();
 * Fan out��create_template();
 * Version��
 *		v1.0	2017.4.19 create by July��the first version
 *---------------------------------------------------------
 */
double compute_matrix_std( cv::Mat src_mat, int flag )
{
	cv::Mat mean_mat(src_mat.rows, src_mat.cols, CV_32FC1 );
	cv::Mat sub_mat(src_mat.rows, src_mat.cols, CV_32FC1 );

	float *pf_data = NULL;
	double d_ret_val = 0;

	if( src_mat.empty() )
	{
		d_ret_val = -1;
		return d_ret_val;
	}

	mean_matrix( src_mat, mean_mat );

	sub_mat = src_mat - mean_mat;
	d_ret_val = 0;

	for( int h = 0; h < src_mat.rows; h++ )
	{
		pf_data = sub_mat.ptr<float>(h);

		for( int w = 0; w < src_mat.cols; w++ )
		{
			d_ret_val += *(pf_data+w) * ( *(pf_data+w) );
		}
	}

	if( 0 == flag )
	{
		d_ret_val /= (src_mat.rows * src_mat.cols - 1);
	}
	else if( 1 == flag )
	{
		d_ret_val /= src_mat.rows * src_mat.cols;
	}

	return (double)sqrtf((float)d_ret_val);

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
void non_max_suppress(cv::Mat src_mat, int n_region_len, double d_threshold, int n_margin, std::vector<cv::Point> &coords_points)
{
	int n_L = n_region_len;
	cv::Point temp_pt;
	int n_h = src_mat.rows;
	int n_w = src_mat.cols;
	float f_gray_val = 0;
	float f_max_val = 0;
	bool b_failed;

	//����ֵ������
	int n_max_x = 0;
	int n_max_y = 0;

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
			//temp_pt.x = c;
			//temp_pt.y = r;

			n_max_x = c;
			n_max_y = r;

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
			for( int r3 = (n_max_y-n_L); r3 < min( (n_max_y+n_L),(n_h-n_margin) ); r3++ )
			{
				for( int c3 = (n_max_x-n_L); c3 < min( (n_max_x+n_L),(n_w-n_margin) ); c3++ )
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
 *		1��cv::Mat					grad_x_img				in		x�᷽����ݶ�ͼ��	
 *		2��cv::Mat					grad_y_img				in		y�᷽����ݶ�ͼ��
 *		3��cv::Mat					angle_img				in		�ݶȷ���ͼ��
 *		4��cv::Mat					weight_img				in		�ݶ�Ȩ��ͼ��
 *		5��std::vector<cv::Point2d>	coords_points			in	    ����������ؼ��ǵ�����
 *		6��std::vector<cv::Point2f>	&corner_subpixel_coords	inout   �����������ؼ��ǵ�����
 *		7��std::vector<cv::Point2f> &corner_v1,				inout	�ǵ㷽���ֵ��cos��sinֵ
 *		8��std::vector<cv::Point2f> &corner_v2				inout	�ǵ㷽���ֵ��cos��sinֵ
 *		9��int						r						in		�ֲ���������߳�
 * Fan in��
 * Fan out��find_corner();
 * Version��
 *		v1.0	2017.5.15 create by July
 *---------------------------------------------------------
 */
void corner_coords_subpixel_refine( cv::Mat grad_x_img, cv::Mat grad_y_img,			\
								        cv::Mat angle_img, cv::Mat weight_img,				\
										std::vector<cv::Point> corner_coords,			\
										std::vector<cv::Point2f> &corner_subpixel_coords, \
										std::vector<cv::Point2f> &corner_v1,				\
										std::vector<cv::Point2f> &corner_v2,int r )
{
	int n_height = grad_x_img.rows;
	int n_width  = grad_x_img.cols;

	cv::Point n_corner_pos_old(0,0);
	cv::Point2f f_corner_pos_new(0,0);

	std::vector<float> v1;
	std::vector<float> v2;
	cv::Point2f temp_pt;

	float f_temp_val = 0;
	float f_temp_val_v1 = 0;
	float f_temp_val_v2 = 0;
	float f_temp_norm = 0;

	int n_ret_flag = -1;

	int cu = 0;
	int cv = 0;

	float f_du = 0;
	float f_dv = 0;

	float f_d1 = 0;
	float f_d2 = 0;

	int start_x = 0; //start_x
	int start_y = 0; //start_y
	int end_x = 0; //end_x
	int end_y = 0; //end_y

	int w = 0;
	int h = 0;

	//�м����
	cv::Mat A1 = Mat::zeros(2,2,CV_32FC1); 
	cv::Mat A2 = Mat::zeros(2,2,CV_32FC1); 

	cv::Mat G  = Mat::zeros(2,2,CV_32FC1);
	cv::Mat H  = Mat::zeros(2,2,CV_32FC1);

	cv::Mat b  = Mat::zeros(2,1,CV_32FC1);

	cv::Mat G_invert = Mat::zeros(2,2,CV_32FC1);

	float *pf_w = new float[2];
	memset( (float*)pf_w, 0, sizeof(float)*2 );

	double *A1_eigen_val = new double[2];  //����A1������ֵ
	double *A1_eigen_vec = new double[4];  //����A1����������
						 
	double *A2_eigen_val = new double[2];  //����A2������ֵ
	double *A2_eigen_vec = new double[4];	//����A2����������

	double *pd_A2 = new double[4]; //����A2
	double *pd_A1 = new double[4]; //����A1

	//�Ծ���A2������ֵ������������ʼ��
	memset((double*)pd_A1, 0, sizeof(double)*4);
	memset((double*)A1_eigen_val, 0, sizeof(double)*2);
	memset((double*)A1_eigen_vec, 0, sizeof(double)*4);

	//�Ծ���A1������ֵ������������ʼ��
	memset((double*)pd_A2, 0, sizeof(double)*4);
	memset((double*)A2_eigen_val, 0, sizeof(double)*2);
	memset((double*)A2_eigen_vec, 0, sizeof(double)*4);

	//���صķ�����������ʱ����
	std::vector<float> o_vec;

	//�ж�������Ĳ����Ƿ���Ϊ�յģ��������������
	if( grad_x_img.empty() || grad_y_img.empty() || angle_img.empty() || weight_img.empty() || corner_coords.empty() )
	{
		return ;
	}

	if( !corner_v1.empty() )
	{
		corner_v1.clear();
	}

	if( !corner_v2.empty() )
	{
		corner_v2.clear();
	}

	if( !corner_subpixel_coords.empty() )
	{
		corner_subpixel_coords.clear();
	}

	//�ǵ㷽�������ʼ��Ϊ0
	for( int i = 0; i < corner_coords.size(); i++ )
	{
		corner_v1.push_back( Point2f(0,0) );
		corner_v2.push_back( Point2f(0,0) );
		corner_subpixel_coords.push_back( Point2f(0,0) );
	}

	if( !o_vec.empty() )
	{
		o_vec.clear();
	}

	o_vec.push_back( 0 );
	o_vec.push_back( 0 );

	//�����еĽǵ�������¾������
	for( int i = 0; i < corner_coords.size(); i++ )
	{
		cu = corner_coords[i].x;
		cv = corner_coords[i].y;
		
		start_x = Max( cu - r, 0);
		start_y = Max( cv - r, 0);

		end_x = Min( cu+r, n_width-1 );
		end_y = Min( cv+r, n_height-1 );

		w = end_x - start_x + 1;
		h = end_y - start_y + 1;

		Rect rect(start_x, start_y, w, h);

		Mat angle_sub = angle_img( rect );		//�õ��Ƕ�ͼ���е�һ����
		Mat weight_sub = weight_img( rect );	//�õ�Ȩ��ͼ���е�һ����

		//����ͼ����б��ط�����㣬���ص����ݱ�����v1��v2��
		n_ret_flag = edge_orientations( angle_sub, weight_sub, v1, v2 );

		temp_pt.x = v1[0];
		temp_pt.y = v1[1];

		corner_v1[i] = temp_pt;

		temp_pt.x = v1[0];
		temp_pt.y = v1[1];

		corner_v2[i] = temp_pt;

		if( (0 == v1[0] && 0 == v1[1]) || (0 == v2[0] && 0 == v2[1]) )
		{
			continue;
		}

		for( int r = start_y; r < end_y; r++ )
		{
			for( int c = start_x; c < end_x; c++ )
			{
				o_vec[0] = grad_x_img.at<float>(r,c) ;
				o_vec[1] = grad_y_img.at<float>(r,c) ;

				if( compute_vec_norm( o_vec ) < 0.1f )
				{
					continue;
				}

				o_vec[0] /= compute_vec_norm( o_vec ) ;
				o_vec[1] /= compute_vec_norm( o_vec ) ;

				f_temp_val = o_vec[0] * v1[0] + o_vec[1] * v1[1];

				//robust refinement of orientation 1
				if( Abs( f_temp_val ) < 0.25 )
				{
					A1.at<float>(0,0) += grad_x_img.at<float>(r,c) * grad_x_img.at<float>(r,c);
					A1.at<float>(0,1) += grad_x_img.at<float>(r,c) * grad_y_img.at<float>(r,c);

					A1.at<float>(1,0) += grad_y_img.at<float>(r,c) * grad_x_img.at<float>(r,c);
					A1.at<float>(1,1) += grad_y_img.at<float>(r,c) * grad_y_img.at<float>(r,c);
				}

				f_temp_val = o_vec[0] * v2[0] + o_vec[1] * v2[1];

				//robust refinement of orientation 2
				if( Abs( f_temp_val ) < 0.25 )
				{
					A2.at<float>(0,0) += grad_x_img.at<float>(r,c) * grad_x_img.at<float>(r,c);
					A2.at<float>(0,1) += grad_x_img.at<float>(r,c) * grad_y_img.at<float>(r,c);

					A2.at<float>(1,0) += grad_y_img.at<float>(r,c) * grad_x_img.at<float>(r,c);
					A2.at<float>(1,1) += grad_y_img.at<float>(r,c) * grad_y_img.at<float>(r,c);
				}

			}// end for( c < w ) inner loop

		}//end for( r < h ) outer loop

		*(pd_A1+0) = A1.at<float>(0,0);
		*(pd_A1+1) = A1.at<float>(0,1);
		*(pd_A1+2) = A1.at<float>(1,0);
		*(pd_A1+3) = A1.at<float>(1,1);

		*(pd_A2+0) = A2.at<float>(0,0);
		*(pd_A2+1) = A2.at<float>(0,1);
		*(pd_A2+2) = A2.at<float>(1,0);
		*(pd_A2+3) = A2.at<float>(1,1);

		//������������ֵ����������
		Eigen_Jacbi( pd_A1, 2, A1_eigen_vec, A1_eigen_val, 0.001, 100 );
		Eigen_Jacbi( pd_A2, 2, A2_eigen_vec, A2_eigen_val, 0.001, 100 );
		
		//���½ǵ�ķ�����Ϣ
		temp_pt.x = (float)*(A1_eigen_vec+0);
		temp_pt.y = (float)*(A1_eigen_vec+2);

		corner_v1[i] = temp_pt;
		v1[0] = temp_pt.x;
		v1[1] = temp_pt.y;

		temp_pt.x = (float)*(A2_eigen_vec+0);
		temp_pt.y = (float)*(A2_eigen_vec+2);

		corner_v2[i] = temp_pt;
		v2[0] = temp_pt.x;
		v2[1] = temp_pt.y;


		for( int r = start_y; r < end_y; r++ )
		{
			for( int c = start_x; c < end_x; c++ )
			{
				o_vec[0] = grad_x_img.at<float>(r,c) ;
				o_vec[1] = grad_y_img.at<float>(r,c) ;

				if( compute_vec_norm( o_vec ) < 0.1f )
				{
					continue;
				}

				o_vec[0] /= compute_vec_norm( o_vec ) ;
				o_vec[1] /= compute_vec_norm( o_vec ) ;

				//robust subpixel corner estimation
				if( r != cv || c != cu )
				{
					//compute rel. position of pixel and distance to vectors
					*(pf_w+0) = c - cu;
					*(pf_w+1) = r - cv;
					
					//����pf_w��v1�����ķ�������ͬmatlab�е�norm
					f_temp_val = *(pf_w+0) * v1[0] + *(pf_w+1) * v1[1] ;
					*(pf_w+0) = *(pf_w+0) - f_temp_val * v1[0];
					*(pf_w+1) = *(pf_w+1) - f_temp_val * v1[1];
					f_d1 = *(pf_w+0) * (*(pf_w+0)) + *(pf_w+1) * (*(pf_w+1));
					f_d1 = sqrtf(f_d1);

					//����pf_w��v2�����ķ�����
					f_temp_val = *(pf_w+0) * v2[0] + *(pf_w+1) * v2[1] ;
					*(pf_w+0) = *(pf_w+0) - f_temp_val * v2[0];
					*(pf_w+1) = *(pf_w+1) - f_temp_val * v2[1];
					f_d1 = *(pf_w+0) * (*(pf_w+0)) + *(pf_w+1) * (*(pf_w+1));
					f_d1 = sqrtf(f_d1);

					f_temp_val_v1 = o_vec[0] * v1[0] + o_vec[1] * v1[1];
					f_temp_val_v2 = o_vec[0] * v2[0] + o_vec[1] * v2[1];

					//if pixel corresponds with either of the vectors / directions
					if( ( f_d1 < 3 && Abs(f_temp_val_v1) < 0.25 ) || ( f_d2 < 3 && Abs(f_temp_val_v2) < 0.25) )
					{
						f_du = grad_x_img.at<float>(r,c);
						f_dv = grad_y_img.at<float>(r,c);

						H.at<float>(0,0) = f_du * f_du;
						H.at<float>(0,1) = f_du * f_dv;
						H.at<float>(1,0) = f_du * f_dv;
						H.at<float>(1,1) = f_dv * f_dv;

						G = G + H;

						b.at<float>(0,0) += H.at<float>(0,0) * c + H.at<float>(0,1) * r;
						b.at<float>(1,0) += H.at<float>(1,0) * c + H.at<float>(1,1) * r;

					}

				} //end if( r != cv || c != cu )

			} //end for( c < end_x ) inner loop

		}//end for( r < end_y ) outer loop

		//�������G���ȣ����f_temp_val������0��
		f_temp_val = G.at<float>(0,0) * G.at<float>(1,1) - G.at<float>(0,1) * G.at<float>(1,0);

		if( Abs( f_temp_val ) > 0.001 )
		{
			n_corner_pos_old = corner_coords[i];

			//�����G������󣬱����Ǽ���G\b = G_inver .* b�ģ�ע�����G�������b
			// ��ʽ���£�
			//
			//		 | a   b |                         1	  | d   -b |
			//	G =  | 	     |	===>  G_invert = ������������*|	       |
			//		 | c   d |				       ad - bc	  | -c   a |
			//
			//-------------------------------------------------------------
			G_invert.at<float>(0,0) = G.at<float>(1,1) / (f_temp_val);
			G_invert.at<float>(0,1) = -1 * G.at<float>(0,1) / (f_temp_val);
			G_invert.at<float>(1,0) = -1 * G.at<float>(1,0) / (f_temp_val);
			G_invert.at<float>(1,1) = G.at<float>(0,0) / (f_temp_val);

			f_corner_pos_new.x = G_invert.at<float>(0,0) * b.at<float>(0,0) + G_invert.at<float>(0,1) * b.at<float>(1,0); 
			f_corner_pos_new.y = G_invert.at<float>(1,0) * b.at<float>(0,0) + G_invert.at<float>(1,1) * b.at<float>(1,0); 

			corner_subpixel_coords[i].x = f_corner_pos_new.x;
			corner_subpixel_coords[i].y = f_corner_pos_new.y;

			f_temp_norm = (f_corner_pos_new.x - n_corner_pos_old.x)*(f_corner_pos_new.x - n_corner_pos_old.x) + \
						  (f_corner_pos_new.y - n_corner_pos_old.y)*(f_corner_pos_new.y - n_corner_pos_old.y);

			f_temp_norm = sqrtf( f_temp_norm );

			if( f_temp_norm > 4 )
			{
				temp_pt.x = 0;
				temp_pt.y = 0;

				corner_v1[i] = temp_pt;
				corner_v2[i] = temp_pt;
			}

		} //end if ( rank(G) == 2 )
		else
		{
			//���½ǵ�ķ�����Ϣ
			temp_pt.x = 0;
			temp_pt.y = 0;

			corner_v1[i] = temp_pt;
			corner_v2[i] = temp_pt;
			
		}


	} // end for(i < corner_coords.size())

	//�ͷ��ڴ沢��ָ��ָ��յ�ַ
	delete []pd_A1;
	delete []pd_A2;
	pd_A1 = NULL;
	pd_A2 = NULL;

	delete []A1_eigen_val;
	delete []A1_eigen_vec;
	A1_eigen_val = NULL;
	A1_eigen_vec = NULL;

	delete []A2_eigen_val;
	delete []A2_eigen_vec;
	A2_eigen_val = NULL;
	A2_eigen_vec = NULL;

	delete []pf_w;
	pf_w = NULL;

	return ;
}

/*
 *---------------------------------------------------------
 * Brief���ǵ�������֣������ݶȵ�Ȩ�غ�x��y����ĽǶȽ�������
 * Return�� float���͵����ֽ��
 * Param��
 *		1��sub_srcImg				in			ԭͼ��	
 *		2��weight_img				in			�ݶ�Ȩ��ͼ��
 *		3��&coords_pts_v1			inout	    �Խǵ㷽��ͳ�Ƶ�һ������
 *		4��&coords_pts_v2			inout		�Խǵ㷽��ͳ�Ƶ���һ������
 * Fan in��score_corner();
 * Fan out��compute_array_std(),compute_array_mean(),compute_array_sum()
 *			compute_array_dot_product(),create_template()
 * Version��
 *		v1.0	2017.4.20 create by July
 *---------------------------------------------------------
 */
float corner_correlation_score( cv::Mat sub_srcImg, cv::Mat weight_img,    \
							       ws_Data_f_2d coords_pts_v1, \
							       ws_Data_f_2d coords_pts_v2 )
{
	float f_score = 0.0f;
	int n_height = weight_img.rows;
	int n_width  = weight_img.cols;

	//Ȩ��ͼ�����ز���
	double d_weight_img_sum = 0;
	double d_weight_img_mean = 0;
	double d_weight_img_std = 0; //��׼��

	//�˲�ͼ�����ز���
	double d_filter_img_sum = 0; //��
	double d_filter_img_mean = 0;//��ֵ
	double d_filter_img_std = 0; //��׼��

	double d_score_grad = 0;
	double d_score_intensity = 0;

	double d_a1 = 0;
	double d_a2 = 0;
	double d_b1 = 0;
	double d_b2 = 0;

	double d_mean = 0;
	double d_score_a = 0;
	double d_score_b = 0;
	double d_score_1 = 0;
	double d_score_2 = 0;
	
	float f_temp_data = 0;
	float f_norm_1 = 0;
	float f_norm_2 = 0;

	//ͼ���ȵ�һ��
	int n_half_width = (int)(n_width+1) / 2 ;
	float *src_img_vec = new float[n_height*n_width];
	memset( (float*)src_img_vec, 0, sizeof(float)*n_height*n_width );
	//center
	float *f_c = new float[2];
	memset( (float*)f_c, n_half_width, sizeof(float)*2 );

	float *f_p1 = new float[2];
	float *f_p2 = new float[2];
	float *f_p3 = new float[2];

	memset( (float*)f_p1, 0, sizeof(float)*2 );
	memset( (float*)f_p2, 0, sizeof(float)*2 );
	memset( (float*)f_p3, 0, sizeof(float)*2 );

	float *img_filter = new float[n_height*n_width];
	memset( (float*)img_filter, -1, sizeof(float)*n_height*n_width );

	float *vec_weight = new float[n_height*n_width];
	memset( (float*)vec_weight, 0, sizeof(float)*n_height*n_width );

	float *dot_product_vec = new float[n_height*n_width];
	memset( (float*)dot_product_vec, 0, sizeof(float)*n_height*n_width );

	//�ó���ģ������
	float *template_a1_vec = new float[n_height*n_width];
	float *template_a2_vec = new float[n_height*n_width];
	float *template_b1_vec = new float[n_height*n_width];
	float *template_b2_vec = new float[n_height*n_width];

	memset( (float*)template_a1_vec, 0, sizeof(float)*n_height*n_width );
	memset( (float*)template_a2_vec, 0, sizeof(float)*n_height*n_width );
	memset( (float*)template_b1_vec, 0, sizeof(float)*n_height*n_width );
	memset( (float*)template_b2_vec, 0, sizeof(float)*n_height*n_width );

	std::vector<cv::Mat> template_vec;


	for( int h = 0; h < n_height; h++ )
	{
		for( int w = 0; w < n_width; w++ )
		{
			*(vec_weight + h * n_width + w) = weight_img.at<float>(h,w);
			*(src_img_vec + h * n_width + w) = sub_srcImg.at<float>(h,w);

			f_p1[0] = h - f_c[0];
			f_p1[1] = w - f_c[0];

			f_temp_data = f_p1[0] * coords_pts_v1.x + f_p1[1] * coords_pts_v1.y;
			f_p2[0] = f_temp_data * coords_pts_v1.x;
			f_p2[1] = f_temp_data * coords_pts_v1.y;

			f_temp_data = f_p1[0] * coords_pts_v2.x + f_p1[1] * coords_pts_v2.y;
			f_p3[0] = f_temp_data * coords_pts_v2.x;
			f_p3[1] = f_temp_data * coords_pts_v2.y;

			f_norm_1 = (f_p1[0] - f_p2[0]) * (f_p1[0] - f_p2[0]) + (f_p1[1] - f_p2[1]) * (f_p1[1] - f_p2[1]);
			f_norm_2 = (f_p1[0] - f_p3[0]) * (f_p1[0] - f_p3[0]) + (f_p1[1] - f_p3[1]) * (f_p1[1] - f_p3[1]);

			if( f_norm_1 <= 1.5 || f_norm_2 <= 1.5 )
			{
				*(img_filter + h * n_width + w) = +1;
			}

		}//end for w

	}//end for h

	//���˲�ͼ����б�׼������
	d_filter_img_sum = compute_array_sum( img_filter, n_height*n_width );
	d_filter_img_mean = d_filter_img_sum / (n_height * n_width ) ;
	d_filter_img_std = compute_array_std( img_filter, n_height*n_width, 0 );

	for( int i = 0; i < n_height*n_width; i++ )
	{
		*(img_filter + i) = ( *(img_filter + i) - d_filter_img_mean ) / d_filter_img_std;
	}

	//��Ȩ��ͼ����б�׼������
	d_weight_img_sum = compute_array_sum( vec_weight, n_height*n_width );
	d_weight_img_mean = d_weight_img_sum / (n_height * n_width);
	d_weight_img_std = compute_array_std( vec_weight, n_height*n_width, 0 );


	for( int i = 0; i < n_height*n_width; i++ )
	{
		*(vec_weight + i) = ( *(vec_weight + i) - d_weight_img_mean ) / d_weight_img_std;
	}

	compute_array_dot_product( vec_weight, img_filter, dot_product_vec, n_height*n_width );

	d_score_grad = Max( compute_array_sum(dot_product_vec, n_height*n_width )/(n_height*n_width-1),0);

	create_template( atan2f(coords_pts_v1.y,coords_pts_v1.x), atan2f(coords_pts_v2.y,coords_pts_v2.x), f_c[0]-1, template_vec );

	for( int h = 0; h < n_height; h++ )
	{
		for( int w = 0; w < n_width; w++ )
		{
			*(template_a1_vec + h * n_width + w) = template_vec[0].at<float>(h,w);
			*(template_a2_vec + h * n_width + w) = template_vec[1].at<float>(h,w);
			*(template_b1_vec + h * n_width + w) = template_vec[2].at<float>(h,w);
			*(template_b2_vec + h * n_width + w) = template_vec[3].at<float>(h,w);

		}//end for w
	}// end for h

	//����ģ��a1��ԭͼ������Ľ��
	memset( (float*)dot_product_vec, 0, sizeof(float)*n_height*n_width );
	compute_array_dot_product( template_a1_vec, src_img_vec, dot_product_vec, n_height*n_width );
	d_a1 = compute_array_sum(dot_product_vec, n_height*n_width );
    //����ģ��a2��ԭͼ������Ľ��
	memset( (float*)dot_product_vec, 0, sizeof(float)*n_height*n_width );
	compute_array_dot_product( template_a2_vec, src_img_vec, dot_product_vec, n_height*n_width );
	d_a2 = compute_array_sum(dot_product_vec, n_height*n_width );

	//����ģ��b1��ԭͼ������Ľ��
	memset( (float*)dot_product_vec, 0, sizeof(float)*n_height*n_width );
	compute_array_dot_product( template_b1_vec, src_img_vec, dot_product_vec, n_height*n_width );
	d_b1 = compute_array_sum(dot_product_vec, n_height*n_width );
    //����ģ��b2��ԭͼ������Ľ��
	memset( (float*)dot_product_vec, 0, sizeof(float)*n_height*n_width );
	compute_array_dot_product( template_b2_vec, src_img_vec, dot_product_vec, n_height*n_width );
	d_b2 = compute_array_sum(dot_product_vec, n_height*n_width );

	d_mean = (d_a1 + d_a2 + d_b1 + d_b2) / 4;

	//case 1: a = white, b = black
	d_score_a = Min( d_a1-d_mean, d_a2-d_mean);
	d_score_b = Min( d_mean-d_b1, d_mean-d_b2 );
	d_score_1 = Min( d_score_a, d_score_b);

	//case 2: b = white, a = black
	d_score_a = Min(d_mean-d_a1, d_mean-d_a2);
	d_score_b = Min(d_b1-d_mean, d_b2-d_mean);
	d_score_1 = Min( d_score_a, d_score_b);

	//intensity score: max. of the 2 cases
	d_score_intensity = Max( Max( d_score_1, d_score_2 ), 0 );

	//final score: product of ggradient and intensity score
	f_score = d_score_grad * d_score_intensity;

	//free memory
	delete []f_c;
	f_c = NULL;

	delete []f_p1;
	delete []f_p2;
	delete []f_p3;

	f_p1 = NULL;
	f_p2 = NULL;
	f_p3 = NULL;

	delete []src_img_vec;
	delete []img_filter;
	delete []vec_weight;

	src_img_vec = NULL;
	img_filter = NULL;
	vec_weight = NULL;

	delete []dot_product_vec;
	delete []template_a1_vec;
	delete []template_a2_vec;
	delete []template_b1_vec;
	delete []template_b2_vec;

	dot_product_vec = NULL;
	template_a1_vec = NULL;
	template_a2_vec = NULL;
	template_b1_vec = NULL;
	template_b2_vec = NULL;

	return f_score;

}

//�Զ���ıȽϺ���,��������
static bool myCompare(const float a1,const float a2)
{
    return a1 <= a2;
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
 * Fan in��corner_coords_subpixel_refine();
 * Fan out��find_modes_meanshift(),
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
	ws_Data_f_3d temp_data;
	std::vector<float> temp_angle_vec;

	std::vector<float> smoothed_hist;
	float delta_angle = 0;


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

	v1.push_back(0);
	v1.push_back(0);

	if( !v2.empty() )
	{
		v2.clear();
	}

	v2.push_back(0);
	v2.push_back(0);

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
	
	//convert angles from normals to directions
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
		temp_angle_vec.push_back( (modes[i].x - 1) * C_PI / n_bin_num );
	}

	sort( temp_angle_vec.begin(), temp_angle_vec.end(), myCompare );

	for( int i = 0; i < 2; i++ )
	{
		//modes_expand
		temp_data.x = modes[i].x;
		temp_data.y = modes[i].y;
		temp_data.z = temp_angle_vec[i];

		modes_expand.push_back( temp_data );

	}

	delta_angle = Min( (modes_expand[1].z - modes_expand[0].z), (modes_expand[0].z + C_PI - modes_expand[1].z) );

	if( delta_angle < 0.3 )
	{
		return -1;
	}

	v1.push_back( cosf(modes_expand[0].z) );
	v1.push_back( sinf(modes_expand[0].z) );

	v2.push_back( cosf(modes_expand[1].z) );
	v2.push_back( sinf(modes_expand[1].z) );

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
 * Fan in�� edge_orientations();
 * Fan out��hist_smooth(),mode_find(),sort_mode()
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
/*
 *---------------------------------------------------------
 * Brief���Խǵ������������
 * Return�� ��
 * Param��	
 *		1��cv::Mat						src_Mat,					in		�����ԭͼ������
 *		2��cv::Mat						angle_img,					in		�Ƕ�ͼ��
 *		3��cv::Mat						weight_img					in		Ȩ��ͼ��
 *		4��std::vector<cv::Point2f>		corner_subpixel_coords,		in		�����ؽǵ�����
 *		5��std::vector<cv::Point2f>		corner_v1,					in		�ǵ�ǶȲ���1
 *		6��std::vector<cv::Point2f>		corner_v2,					in		�ǵ�ǶȲ���2
 *		7��std::vector<int>				radius, 					in		�߶Ȱ뾶����
 *		8��std::vector<float>			&score_corner_table			inout	�õ�����������
 * Fan in��find_corner();
 * Fan out��corner_correlation_score(),round()
 * Version��
 *		v1.0	2017.5.17 create by July
 *---------------------------------------------------------
 */
void score_corner( cv::Mat src_Mat, cv::Mat angle_img, cv::Mat weight_img, 
					std::vector<cv::Point2f> corner_subpixel_coords, 
					std::vector<cv::Point2f> corner_v1,
					std::vector<cv::Point2f> corner_v2,
				    std::vector<int> radius, 
					std::vector<float> &score_corner_table )
{
	int n_height = 0;
	int n_width  = 0;

	int n_x = 0;
	int n_y = 0;

	int start_x = 0; //start_x
	int start_y = 0; //start_y
	int end_x = 0; //end_x
	int end_y = 0; //end_y

	int w = 0;
	int h = 0;

	float f_max_val = 0;
	float f_min_val = 9999;

	ws_Data_f_2d coords_pts_v1;
	ws_Data_f_2d coords_pts_v2;

	float *pf_score = NULL;

	if( src_Mat.empty() || angle_img.empty() || weight_img.empty() ||
		corner_subpixel_coords.empty() || corner_v1.empty() ||
		corner_v2.empty() || radius.empty() )
	{
		if( !score_corner_table.empty() )
		{
			score_corner_table.clear();
		}
		return ;
	}

	n_height = src_Mat.rows;
	n_width  = src_Mat.cols;

	pf_score = new float[radius.size()];
	memset( (float*)pf_score, 0, sizeof(float)*radius.size() );

	//for all corner do
	for( int i = 0; i < corner_subpixel_coords.size(); i++ )
	{
		//n_x = (int)(corner_subpixel_coords[i].x + 0.5);
		n_x = round(corner_subpixel_coords[i].x);
		n_y = round(corner_subpixel_coords[i].y);
		
		for( int j = 0; j < radius.size(); j++ )
		{
			*(pf_score+j) = 0;

			if( n_x >= radius[j] && n_x < n_width-radius[j] &&
				n_y >= radius[j] && n_y < n_height-radius[j] )
			{
				start_x = n_x - radius[j];
				start_y = n_y - radius[j];

				end_x = n_x + radius[j];
				end_y = n_y + radius[j];

				w = end_x - start_x + 1;
				h = end_y - start_y + 1;

				Rect rect(start_x, start_y, w, h);
				Mat src_img_sub = src_Mat( rect );     //�õ�ԭʼ�Ҷ�ͼ���е�һ����
				Mat angle_sub = angle_img( rect );		//�õ��Ƕ�ͼ���е�һ����
				Mat weight_sub = weight_img( rect );	//�õ�Ȩ��ͼ���е�һ����

				coords_pts_v1.x = corner_v1[i].x;
				coords_pts_v1.y = corner_v1[i].y;

				coords_pts_v2.x = corner_v2[i].x;
				coords_pts_v2.y = corner_v2[i].y;

				*(pf_score+j) =  corner_correlation_score( src_img_sub, weight_sub, coords_pts_v1, coords_pts_v2 ); 

			} //end if

		} //end for j < radius.size()

		f_max_val = 0;
		f_min_val = 9999;

		find_array_max_min_val( pf_score, radius.size(), &f_max_val, &f_min_val ); 

		score_corner_table.push_back(f_max_val);

	}// end for i < corner_subpixel_coords.size()

	delete []pf_score;
	pf_score = NULL;

	return ;

}

/*
 *---------------------------------------------------------
 * Brief�����ؾ���f_data���������
 * Return�� int���͵�����
 * Param��
 *		1��float f_data	in	��������	
 * Fan in��score_corner()
 * Fan out��
 * Version��
 *		v1.0	2017.5.16 create by July
 *---------------------------------------------------------
 */
int round(float f_data)
{
	int n_ret_val = 0;

	if( f_data > 0 )
	{
		return n_ret_val = (int)(f_data+0.5);
	}
	else if( f_data < 0 )
	{
		return n_ret_val = (int)(f_data-0.5);
	}

	return n_ret_val;

}

/*
 *---------------------------------------------------------
 * Brief���������������ֵ����Сֵ
 * Return�� ��
 * Param��
 *		1��float *src_data	in		��������������
 *		2��int   n_size     in		����Ĵ�С
 *		3��float *f_max_val inout   ���������ֵ
 *		4��float *f_min_val inout   ��������Сֵ
 * Fan in��score_corner()
 * Fan out��
 * Version��
 *		v1.0	2017.5.16 create by July
 *---------------------------------------------------------
 */
void find_array_max_min_val( float *src_data, int n_size, float *f_max_val, float *f_min_val )
{

	for( int i = 0; i < n_size; i++ )
	{
		if( *(src_data+i) > *f_max_val )
		{
			*f_max_val = *(src_data+i);
		}

		if( *(src_data+i) < *f_min_val )
		{
			*f_min_val = *(src_data+i);
		} 

	}// end for i < n_size;
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

  
/*
 *---------------------------------------------------------
 * Brief:	��ʵ�Գƾ��������ֵ�������������ſ˱ȷ�  
 *			�����Ÿ��(Jacobi)������ʵ�Գƾ����ȫ������ֵ����������
 * Return�� bool ��
 * Param��
 *		1��double	*pMatrix			in          ����Ϊn*n�����飬���ʵ�Գƾ��� 
 *		2��int		nDim                in			����Ľ���n  
 *		3��double	*pdblVects          inout		����Ϊn*n�����飬������������(���д洢) 
 *		4��double	*pdbEigenValues		inout		����ֵ����
 *		5��double	dbEps               in			����Ҫ��  
 *		6��int		nJt                 in		   ���ͱ�������������������            
 * Fan in��corner_coords_subpixel_refine()
 * Fan out��
 * Version��
 *		v1.0	2017.5.15 create by July
 *---------------------------------------------------------
 */

bool Eigen_Jacbi(double * pMatrix,int nDim, double *pdblVects, double *pdbEigenValues, double dbEps,int nJt)  
{  
    for(int i = 0; i < nDim; i ++)   
    {     
        pdblVects[i*nDim+i] = 1.0f;   
        for(int j = 0; j < nDim; j ++)   
        {   
            if(i != j)     
                pdblVects[i*nDim+j]=0.0f;   
        }   
    }   
  
    int nCount = 0;     //��������  
	double temp_val = 0;

	double *temp_vec = new double[2*2];
	memset( (double*)temp_vec, 0, sizeof(double)*4 );

	*(temp_vec + 0) = *( pMatrix + 0 );
	*(temp_vec + 1) = *( pMatrix + 1 );
	*(temp_vec + 2) = *( pMatrix + 2 );
	*(temp_vec + 3) = *( pMatrix + 3 );

    while(1)  
    {  
        //��pMatrix�ķǶԽ������ҵ����Ԫ��  
        double dbMax = pMatrix[1];  
        int nRow = 0;  
        int nCol = 1;  
        for (int i = 0; i < nDim; i ++)          //��  
        {  
            for (int j = 0; j < nDim; j ++)      //��  
            {  
                double d = fabs(pMatrix[i*nDim+j]);   
  
                if((i!=j) && (d> dbMax))   
                {   
                    dbMax = d;     
                    nRow = i;     
                    nCol = j;   
                }   
            }  
        }  
  
        if(dbMax < dbEps)     //���ȷ���Ҫ��   
            break;    
  
        if(nCount > nJt)       //����������������  
            break;  
  
        nCount++;  
  
        double dbApp = pMatrix[nRow*nDim+nRow];  
        double dbApq = pMatrix[nRow*nDim+nCol];  
        double dbAqq = pMatrix[nCol*nDim+nCol];  
  
        //������ת�Ƕ�  
        double dbAngle = 0.5*atan2(-2*dbApq,dbAqq-dbApp);  
        double dbSinTheta = sin(dbAngle);  
        double dbCosTheta = cos(dbAngle);  
        double dbSin2Theta = sin(2*dbAngle);  
        double dbCos2Theta = cos(2*dbAngle);  
  
        pMatrix[nRow*nDim+nRow] = dbApp*dbCosTheta*dbCosTheta +   
            dbAqq*dbSinTheta*dbSinTheta + 2*dbApq*dbCosTheta*dbSinTheta;  
        pMatrix[nCol*nDim+nCol] = dbApp*dbSinTheta*dbSinTheta +   
            dbAqq*dbCosTheta*dbCosTheta - 2*dbApq*dbCosTheta*dbSinTheta;  
        pMatrix[nRow*nDim+nCol] = 0.5*(dbAqq-dbApp)*dbSin2Theta + dbApq*dbCos2Theta;  
        pMatrix[nCol*nDim+nRow] = pMatrix[nRow*nDim+nCol];  
  
        for(int i = 0; i < nDim; i ++)   
        {   
            if((i!=nCol) && (i!=nRow))   
            {   
                int u = i*nDim + nRow;  //p    
                int w = i*nDim + nCol;  //q  
                dbMax = pMatrix[u];   
                pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta;   
                pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta;   
            }   
        }   
  
        for (int j = 0; j < nDim; j ++)  
        {  
            if((j!=nCol) && (j!=nRow))   
            {   
                int u = nRow*nDim + j;  //p  
                int w = nCol*nDim + j;  //q  
                dbMax = pMatrix[u];   
                pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta;   
                pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta;   
            }   
        }  
  
        //������������  
        for(int i = 0; i < nDim; i ++)   
        {   
            int u = i*nDim + nRow;      //p     
            int w = i*nDim + nCol;      //q  
            dbMax = pdblVects[u];   
            pdblVects[u] = pdblVects[w]*dbSinTheta + dbMax*dbCosTheta;   
            pdblVects[w] = pdblVects[w]*dbCosTheta - dbMax*dbSinTheta;   
        }   
  
    }  
  
    //������ֵ���������Լ�����������������,����ֵ��pMatrix���Խ����ϵ�Ԫ��  
    std::map<double,int> mapEigen;  
    for(int i = 0; i < nDim; i ++)   
    {     
        pdbEigenValues[i] = pMatrix[i*nDim+i];  
  
        mapEigen.insert(make_pair( pdbEigenValues[i],i ) );  
    }   
  
    double *pdbTmpVec = new double[nDim*nDim];  
    std::map<double,int>::reverse_iterator iter = mapEigen.rbegin();  

    for (int j = 0; iter != mapEigen.rend(),j < nDim; ++iter,++j)  
    {  
        for (int i = 0; i < nDim; i ++)  
        {  
            pdbTmpVec[i*nDim+j] = pdblVects[i*nDim + iter->second];  
        }  
  
        //����ֵ��������  
        pdbEigenValues[j] = iter->first;  
    }  
  
    //�趨������  
    for(int i = 0; i < nDim; i ++)   
    {  
        double dSumVec = 0;  
        for(int j = 0; j < nDim; j ++)  
            dSumVec += pdbTmpVec[j * nDim + i];  
        if(dSumVec<0)  
        {  
            for(int j = 0;j < nDim; j ++)  
                pdbTmpVec[j * nDim + i] *= -1;  
        }  
    }  

	if( 2 == nDim )
	{
		if( ( *(pdbEigenValues + 0 ) - *(pdbEigenValues + 1 ) ) > 0.001 )
		{
			temp_val = *( pdbEigenValues + 1 );
			*( pdbEigenValues + 1 ) = *( pdbEigenValues + 0 );
			*( pdbEigenValues + 0 ) = temp_val;
		}

		if( *(pdbTmpVec+1) > 0 )
		{
			temp_val = *( pdbTmpVec+0 );
			*( pdbTmpVec+0 ) = Abs( *( pdbTmpVec+1 ) );
			*( pdbTmpVec+1 ) = Abs( temp_val );

			temp_val = *( pdbTmpVec+3 );
			*( pdbTmpVec+3 ) = Abs( *( pdbTmpVec+2 ) ) * (-1);
			*( pdbTmpVec+2 ) = Abs( temp_val );

		}
		else if( *(pdbTmpVec+1) < 0 )
		{
			temp_val = *( pdbTmpVec+0 );
			*( pdbTmpVec+0 ) = Abs( *( pdbTmpVec+1 ) );
			*( pdbTmpVec+1 ) = Abs( temp_val ) * (-1);

			temp_val = *( pdbTmpVec+3 );
			*( pdbTmpVec+3 ) = Abs( *( pdbTmpVec+2 ) ) * (-1);
			*( pdbTmpVec+2 ) = Abs( temp_val ) * (-1);

		}

		temp_val = ( *( temp_vec+0 ) ) * ( *( pdbTmpVec+0 ) ) + ( *( temp_vec+1 ) ) * ( *( pdbTmpVec+2 ) );

		if( Abs( temp_val - *( pdbTmpVec+0 ) * ( *(pdbEigenValues + 0 ) )  ) > 0.001 )
		{
			*( pdbTmpVec+0 ) = *( pdbTmpVec+0 ) * (-1) ;
			*( pdbTmpVec+3 ) = *( pdbTmpVec+3 ) * (-1) ;
		}
	}
  
    memcpy(pdblVects,pdbTmpVec,sizeof(double)*nDim*nDim);  
    delete []pdbTmpVec;  
	pdbTmpVec = NULL;

	delete []temp_vec;
	temp_vec = NULL;
  
    return 1;  
}  

int Jacobi(double matrix[][2], double vec[][2], int maxt, int n)  
{  
   
	int it, p, q, i, j; // ��������ֵ  
	double temp, t, cn, sn, max_element, vip, viq, aip, aiq, apj, aqj; // ��ʱ����  
	for (it = 0; it < maxt; ++it)  
	{  
		max_element = 0;  
		for (p = 0; p < n-1; ++p)  
		{  
			for (q = p + 1; q < n; ++q)  
			{  
				if (fabs(matrix[p][q]) > max_element) // ��¼�ǶԽ���Ԫ�����ֵ  
					max_element = fabs(matrix[p][q]);  
				if (fabs(matrix[p][q]) > EPS) // �ǶԽ���Ԫ�ط�0ʱ��ִ��Jacobi�任  
				{  
					// ����Givens��ת�������ҪԪ��:cos(theta), ��cn, sin(theta), ��sn  
					temp = (matrix[q][q] - matrix[p][p]) / matrix[p][q] / 2.0;  
					if (temp >= 0) // tΪ���� t^2 + 2*t*temp - 1 = 0�ĸ�, ȡ����ֵ��С�ĸ�Ϊt  
						t = -temp + sqrt(1 + temp * temp);  
					else  
						t = -temp - sqrt(1 + temp * temp);  
					cn = 1 / sqrt(1 + t * t);  
					sn = t * cn;  
					// Givens��ת����֮ת�����matrix, ������matrix��p�к�q��  
					for (j = 0; j < n; ++j)  
					{  
						apj = matrix[p][j];  
						aqj = matrix[q][j];  
						matrix[p][j] = cn * apj - sn * aqj;  
						matrix[q][j] = sn * apj + cn * aqj;  
					}  
					// Givens��ת�����ҳ�matrix, ������matrix��p�к�q��  
					for (i = 0; i < n; ++i)  
					{  
						aip = matrix[i][p];  
						aiq = matrix[i][q];  
						matrix[i][p] = cn * aip - sn * aiq;  
						matrix[i][q] = sn * aip + cn * aiq;  
					}  
					// �������������洢����vec, vec=J0��J1��J2...��Jit, ����ÿ��ֻ����vec��p, q����  
					for (i = 0; i < n; ++i)  
					{  
						vip = vec[i][p];  
						viq = vec[i][q];  
						vec[i][p] = cn * vip - sn * viq;  
						vec[i][q] = sn * vip + cn * viq;  
					}  
				}    
			}   
		}   
		if (max_element <= EPS) // �ǶԽ���Ԫ����С��������׼����������  
			return 1;  
	}  
	return 0;  
}

/*
 *---------------------------------------------------------
 * Brief��������������ĵ���� Y = X .* Z
 *		  y1=x1*z1, y2=x2*z2, y3=x3*z3, ..... yn=xn*zn, nΪ����Ĵ�С
 * Return�� double���͵������
 * Param��
 *		1��float *input_array1	in		�������������1
 *		2��float *input_array2	in		�������������2
 *		3��float *output_array  inout	�������������
 *		4��int   n_size			in	����Ĵ�С
 * Fan in��;
 * Fan out��corner_correlation_score();
 * Version��
 *		v1.0	2017.5.16 create by July��the first version
 *---------------------------------------------------------
 */
int compute_array_dot_product( float *input_array1, float *input_array2, float *output_array, int n_array_size )
{
	int n_size = n_array_size;

	if( NULL == input_array1 || NULL == input_array2 || NULL == output_array )
	{
		return -1;
	}

	for( int i = 0; i < n_size; i++ )
	{
		*(output_array + i) = *(input_array1 + i) * ( *(input_array2 + i) );
	}

	return 0;
}
