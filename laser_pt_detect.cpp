#include "stdafx.h"
#include "laser_pt_detect.h"

LaserPointPos::LaserPointPos()
{
	m_laser_point_coords = Point(0,0);
	m_adjust_laser_pt = Point2f(0,0);
	m_norm_img_laser_pt = Point(0,0);

	o_min_bounding_rect.height = -1;
	o_min_bounding_rect.width  = -1;
	o_min_bounding_rect.x = -1;
	o_min_bounding_rect.y = -1;

	o_roi_rect_size.height = -1;
	o_roi_rect_size.width  = -1;

	o_roi_rect_center.x = -1;
	o_roi_rect_center.y = -1;
	count = 0;

}

LaserPointPos::~LaserPointPos()
{
	om_roi_points.clear();
	om_dst_img.release();                    //用矩形框画出激光点的彩色图像
	om_roi_img.release();                     //检测出来的roi区域图像，即从原图像上把roi区域抠出来
	om_adjust_roi_img.release();            //经过校正的roi区域图像，即图像中只有roi区域，且已经做过形状校正处理
	om_adjust_roi_bin_img.release();       //经过校正的roi区域二值图像
	om_roi_bin_img.release();
	om_norm_roi_img.release();
}

/*---------------------------------------------------------------------------
 *
 * function : 激光点检测算法的执行函数
 * 
 * parameter: 
 *	   1、 void * p_data   : 图像数据指针
 *	   2、 int    n_width  : 图像的宽
 *	   3、 int    n_height : 图像的高
 *	   4、 int    &x       : 激光点在校正并规范化后的roi区域图像中的X轴坐标
 *     5、 int    &y       : 激光点在校正并规范化后的roi区域图像中的Y轴坐标
 *     6   laser_type type : 激光点的类型，即激光的颜色
 *---------------------------------------------------------------------------
 */
int LaserPointPos::m_laser_detect_execute( void *p_data, int n_width, int n_height, float &x, float &y, laser_type type )
{
	if( NULL == p_data ){
		cout << "image data is empty!" << endl;
		return -1;
	}

	if( n_height < 0 || n_width < 0 ){
		cout << "the size of the image is smaller than zero!" << endl;
		return -2;
	}

	Mat o_src_img(n_height, n_width, CV_8UC3, p_data );
	Mat o_temp_img( n_height, n_width, CV_8UC3 );
	o_src_img.copyTo( o_temp_img );
	//o_copy_img = o_src_img.clone();
	cvtColor( o_src_img, o_temp_img, CV_RGB2BGR );
	//cv::resize( o_temp_img, o_temp_img, Size( int(n_width/2), int(n_height/2) ), 0, 0, CV_INTER_AREA );
	m_roi_detect( o_temp_img, type );
	//memcpy( p_data, om_norm_roi_img.data, sizeof(n_width*n_height) );
	//p_data = om_adjust_roi_img.data;
	x = m_adjust_laser_pt.x;
	y = m_adjust_laser_pt.y;

	return 0;

}

/*---------------------------------------------------------------------------
 *
 * function : 激光点检测算法的执行函数重载
 * 
 * parameter: 
 *	   1、 Mat o_input_img : 图像数据指针
 *	   2、 int    &x       : 激光点在校正并规范化后的roi区域图像中的X轴坐标
 *     3、 int    &y       : 激光点在校正并规范化后的roi区域图像中的Y轴坐标
 *     4、 laser_type type : 激光类型，即激光的颜色
 *---------------------------------------------------------------------------
 */
int LaserPointPos::m_laser_detect_execute( cv::Mat o_input_img, float &x, float &y, laser_type type )
{
	if( o_input_img.empty() ){
		cout << "the input image of the function m_laser_detect_execute() is empty!" << endl;
		return -1;
	}

	Mat o_src_img;
	o_input_img.copyTo(o_src_img);
	m_roi_detect( o_src_img, type );
	x = m_adjust_laser_pt.x;
	y = m_adjust_laser_pt.y;

	return 0;
}

/*-----------------------------------------------------------
 *
 * function : 激光点检测算法
 * 
 * parameter: 
 *		Mat o_intput_img  输入的原图像
 *-------------------------------------------------------------
 */
int LaserPointPos::m_laser_point_detect( Mat o_input_img )
{
	if( o_input_img.empty() ){
		cout << "the input image of the funtion m_laser_point_detect() is empty!" << endl;
		return -1;
	}

	m_laser_point_coords = Point(0,0);

 	Mat o_bin_img(o_input_img.rows, o_input_img.cols, CV_8UC1 );
	o_bin_img = Scalar::all(0);
	//cv::cvtColor( o_input_img, o_input_img, CV_BGR2RGB );
	Mat o_src_img;
	o_input_img.copyTo(o_src_img);
	//namedWindow("src_img1", 0 );
	//imshow("src_img1", o_src_img );
	//waitKey(0);
	//m_roi_detect( o_src_img );


	//blur( o_src_img, o_src_img, Size(5,5) );
	//medianBlur( o_src_img, o_src_img, 3 );
	vector<Mat> o_channels;
	Mat o_r_channel;
	Mat o_g_channel;
	Mat o_b_channel;

	split( o_src_img, o_channels );
	o_b_channel = o_channels.at(0);
	o_g_channel = o_channels.at(1);
	o_r_channel = o_channels.at(2);

	int n_height = o_input_img.rows;
	int n_width  = o_input_img.cols;
	uchar *p;
	double d_min_val = 0;
	double d_max_val = 0;
	Point o_min_val_pt;
	Point o_max_val_pt;
	//minMaxLoc( o_r_channel, &d_min_val, &d_max_val, &o_min_val_pt, &o_max_val_pt );
	
	int n_max_val = 0;
	int n_min_val = 65535;
	int n_max_sub_val = 0;

	//********************************************************************
	/*
	 * 考虑到红色激光点照射的地方肯定是 R 值较大的或者是最大的点
	 * 为了滤除噪声和白色像素点，对 G、B 通道进行大小限制
	 * 得到 R 通道中 最大值
	 */
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
			if(  o_r_channel.at<uchar>(row, col) > n_max_val && ( o_g_channel.at<uchar>(row, col) < 175 || o_b_channel.at<uchar>(row, col) < 175 ) ){ //150
					n_max_val = o_r_channel.at<uchar>(row, col);
			}
		}
	}
	//*************************************************************************

	vector<int> o_row_pts;
	vector<int> o_col_pts;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * 检测激光点，即 R 通道在最大值附近，且 R 分别和 G、B 通道的差值大于一定的值
	 */
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
#if 0
			if( n_max_val > 240 ){
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && o_g_channel.at<uchar>(row, col) < 70 && o_b_channel.at<uchar>(row, col) < 70  ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}else{
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && ( o_r_channel.at<uchar>(row, col) - o_g_channel.at<uchar>(row, col) > 70 || 
					o_r_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}
#else
			if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 2) && ( o_r_channel.at<uchar>(row, col) - o_g_channel.at<uchar>(row, col) > 70 || 
				o_r_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
					o_bin_img.at<uchar>(row,col) = 255;
					
			}

#endif
			
		}
	}

	//FILE *fp;
	//fp = fopen("test.txt", "w" );
#if 1
	//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int n_max_sub_val_RG = 0; // R 分量和 G 分量的最大差值
	int n_max_sub_val_RB = 0; // R 分量和 B 分量的最大差值
	//…………………………………………………………………………………………………
	/*
	 * 寻找候选目标点中， R 分量和 G 分量的最大差值，R 分量和 B 分量的最大差值
	 */
	for( int h = 0; h < n_height; h++ ){
		for( int w = 0; w < n_width; w++ ){
			if( o_bin_img.at<uchar>( h, w ) == 255 ){
				//fprintf(fp,"(x,y) = (%d,%d)\t",w, h );
				//fprintf(fp, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( h, w ), o_g_channel.at<uchar>( h, w ), o_b_channel.at<uchar>( h, w ) );

				if( ( o_r_channel.at<uchar>( h, w ) -  o_g_channel.at<uchar>( h, w ) ) > n_max_sub_val_RG ){
					n_max_sub_val_RG = int( o_r_channel.at<uchar>( h, w ) - o_g_channel.at<uchar>( h, w ) );
				}

				if( ( o_r_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) ) > n_max_sub_val_RB ){
					n_max_sub_val_RB = int(  o_r_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) );
				}

			}else{
				continue;
			}
		}
	}
	//fclose(fp);
	for( int r = 0; r < n_height; r++ ){
		for( int c = 0; c < n_width; c++ ){
			if( o_bin_img.at<uchar>( r, c ) != 0  ){
				if( ( o_r_channel.at<uchar>( r, c ) -  o_g_channel.at<uchar>( r, c ) ) >= (n_max_sub_val_RG - 2 ) ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}

			}else{
				continue;
			}
		}
	}
#else

#endif
	//…………………………………………………………………………………………………
#if 0
	FILE *fp1 = fopen("result.txt", "w" );

	for( int hei = 0; hei < n_height; hei++ ){
		for( int wid = 0; wid < n_width; wid++ ){
			if( o_bin_img.at<uchar>( hei, wid ) != 0 ){
				fprintf(fp1,"(x,y) = (%d,%d)\t",wid, hei );
				fprintf(fp1, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( hei, wid ), o_g_channel.at<uchar>( hei, wid ), o_b_channel.at<uchar>( hei, wid ) );
			}
		}
	}
	fclose(fp1);
#endif

	int n_x_mean_pt = 0;
	int n_y_mean_pt = 0;
	for( int i = 0; i < o_row_pts.size(); i++ ){
		n_y_mean_pt += o_row_pts[i];
		n_x_mean_pt += o_col_pts[i];
	}

	if( !o_row_pts.empty() && !o_col_pts.empty() ){
		n_x_mean_pt /= o_col_pts.size();
		n_y_mean_pt /= o_row_pts.size();
	}else{
		n_x_mean_pt = 0;
		n_y_mean_pt = 0;
	}
	//--------------------在图像上用小的正方形画出激光点--------------------
	Point o_left_top;
	Point o_right_down;
	const int n_offset = 10;
	if( n_x_mean_pt - n_offset >= 0 ){
		o_left_top.x = n_x_mean_pt - n_offset;
	}else{
		o_left_top.x = 0;
	}

	if(n_y_mean_pt - n_offset >= 0){
		o_left_top.y = n_y_mean_pt - n_offset;
	}else{
		o_left_top.y = 0;
	}

	if( n_x_mean_pt + n_offset <= n_width - 1){
		o_right_down.x = n_x_mean_pt + n_offset;
	}else{
		o_right_down.x = n_width - 1;
	}

	if( n_y_mean_pt + n_offset <= n_height - 1){
		o_right_down.y = n_y_mean_pt + n_offset;
	}else{
		o_right_down.y = n_height - 1;
	}
	
	rectangle( o_src_img, o_left_top, o_right_down, CV_RGB(0, 255, 0), 4 );

	//----------------------------------------------------------------------------
	o_src_img.copyTo(om_dst_img);
	m_laser_point_coords = Point( n_x_mean_pt, n_y_mean_pt );

	/*namedWindow("src_img", 0);
	imshow("src_img", o_src_img );
	namedWindow("bin_img", 0 );
	imshow("bin_img", o_bin_img );
	waitKey(0);

	destroyAllWindows();*/

	return 0;
}

/*-----------------------------------------------------------
 *
 * function : 激光点检测算法
 * 
 * parameter: 
 *		Mat o_intput_img  输入的原图像
 *-------------------------------------------------------------
 */
int LaserPointPos::m_laser_point_detect1( Mat o_input_img )
{
	if( o_input_img.empty() ){
		cout << "the input image of the funtion m_laser_point_detect1() is empty!" << endl;
		return -1;
	}

	m_laser_point_coords = Point(0,0);

 	Mat o_bin_img(o_input_img.rows, o_input_img.cols, CV_8UC1 );
	o_bin_img = Scalar::all(0);
	//cv::cvtColor( o_input_img, o_input_img, CV_BGR2RGB );
	Mat o_src_img;
	o_input_img.copyTo(o_src_img);
	//namedWindow("src_img1", 0 );
	//imshow("src_img1", o_src_img );
	//waitKey(0);
	//m_roi_detect( o_src_img );


	//blur( o_src_img, o_src_img, Size(5,5) );
	//medianBlur( o_src_img, o_src_img, 3 );
	vector<Mat> o_channels;
	Mat o_r_channel;
	Mat o_g_channel;
	Mat o_b_channel;

	split( o_src_img, o_channels );
	o_b_channel = o_channels.at(0);
	o_g_channel = o_channels.at(1);
	o_r_channel = o_channels.at(2);


	int n_height = o_input_img.rows;
	int n_width  = o_input_img.cols;

	for( int h = 0; h < n_height; ++h ){
		for( int w = 0; w < n_width; ++w ){
			o_b_channel.at<uchar>( h, w ) = o_b_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
			o_g_channel.at<uchar>( h, w ) = o_g_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
			o_r_channel.at<uchar>( h, w ) = o_r_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
		}
	}

	uchar *p;
	double d_min_val = 0;
	double d_max_val = 0;
	Point o_min_val_pt;
	Point o_max_val_pt;
	//minMaxLoc( o_r_channel, &d_min_val, &d_max_val, &o_min_val_pt, &o_max_val_pt );
	
	int n_max_val = 0;
	int n_min_val = 65535;
	int n_max_sub_val = 0;

	//********************************************************************
	/*
	 * 考虑到红色激光点照射的地方肯定是 R 值较大的或者是最大的点
	 * 为了滤除噪声和白色像素点，对 G、B 通道进行大小限制
	 * 得到 R 通道中 最大值
	 */
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
			if(  o_r_channel.at<uchar>(row, col) > n_max_val && ( o_g_channel.at<uchar>(row, col) < 175 || o_b_channel.at<uchar>(row, col) < 175 ) ){ //150
					n_max_val = o_r_channel.at<uchar>(row, col);
			}
		}
	}
	//*************************************************************************

	vector<int> o_row_pts;
	vector<int> o_col_pts;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * 检测激光点，即 R 通道在最大值附近，且 R 分别和 G、B 通道的差值大于一定的值
	 */
	
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
#if 0
			if( n_max_val > 240 ){
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && o_g_channel.at<uchar>(row, col) < 70 && o_b_channel.at<uchar>(row, col) < 70  ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}else{
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && ( o_r_channel.at<uchar>(row, col) - o_g_channel.at<uchar>(row, col) > 70 || 
					o_r_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}
#else
			if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 2) && ( o_r_channel.at<uchar>(row, col) - o_g_channel.at<uchar>(row, col) > 70 || 
				o_r_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
					o_bin_img.at<uchar>(row,col) = 255;
					
			}

#endif
			
		}
	}
	
	//FILE *fp;
	//fp = fopen("test.txt", "w" );
#if 1
	//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int n_max_sub_val_RG = 0; // R 分量和 G 分量的最大差值
	int n_max_sub_val_RB = 0; // R 分量和 B 分量的最大差值
	//…………………………………………………………………………………………………
	/*
	 * 寻找候选目标点中， R 分量和 G 分量的最大差值，R 分量和 B 分量的最大差值
	 */
	for( int h = 0; h < n_height; h++ ){
		for( int w = 0; w < n_width; w++ ){
			if( o_bin_img.at<uchar>( h, w ) == 255 ){
				//fprintf(fp,"(x,y) = (%d,%d)\t",w, h );
				//fprintf(fp, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( h, w ), o_g_channel.at<uchar>( h, w ), o_b_channel.at<uchar>( h, w ) );

				if( ( o_r_channel.at<uchar>( h, w ) -  o_g_channel.at<uchar>( h, w ) ) > n_max_sub_val_RG ){
					n_max_sub_val_RG = int( o_r_channel.at<uchar>( h, w ) - o_g_channel.at<uchar>( h, w ) );
				}

				if( ( o_r_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) ) > n_max_sub_val_RB ){
					n_max_sub_val_RB = int(  o_r_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) );
				}

			}else{
				continue;
			}
		}
	}
	//fclose(fp);
	for( int r = 0; r < n_height; r++ ){
		for( int c = 0; c < n_width; c++ ){
			if( o_bin_img.at<uchar>( r, c ) != 0  ){
				if( ( o_r_channel.at<uchar>( r, c ) -  o_g_channel.at<uchar>( r, c ) ) >= (n_max_sub_val_RG - 2 ) ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}

			}else{
				continue;
			}
		}
	}
#else

#endif

	Rect o_max_rect;
	bool b_flag = true;
	if( o_row_pts.empty() || o_col_pts.empty() ){
		for( int r = 0; r < n_height; r++ ){
			for( int c = 0; c < n_width; c++ ){
				
				if(  o_r_channel.at<uchar>( r, c ) >= 195 && o_g_channel.at<uchar>( r, c ) >= 180 && o_b_channel.at<uchar>(r, c) >= 170  ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					o_bin_img.at<uchar>( r, c ) = 255;
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}

				
			}
		}
	
		vector<vector<Point>> o_contours;

		findContours( o_bin_img, o_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

		if( o_contours.empty() ){ //如果没有找到候选激光点区域，就默认投影区域中最亮的像素点就是激光点
			b_flag = false;
			int n_max_sum = 0;
			int n_temp_val = 0;
			Point o_max_pt = Point(0,0);
			for( int r = 0; r < n_height; r++ ){
				for( int c = 0; c < n_width; c++ ){

					n_temp_val = o_r_channel.at<uchar>( r, c ) + o_g_channel.at<uchar>( r, c ) + o_b_channel.at<uchar>(r, c) ;
					if( n_temp_val > n_max_sum ){
						o_max_pt.x = c;
						o_max_pt.y = r;
					}else{
						o_bin_img.at<uchar>( r, c ) = 0;
					}
					/*if(  o_r_channel.at<uchar>( r, c ) >= 190 && o_g_channel.at<uchar>( r, c ) >= 170 && o_b_channel.at<uchar>(r, c) >= 170  ){
						o_row_pts.push_back(r);
						o_col_pts.push_back(c);
						o_bin_img.at<uchar>( r, c ) = 255;
						continue;
					}else{
						o_bin_img.at<uchar>( r, c ) = 0;
					}*/


				}
			}

			o_row_pts.push_back(o_max_pt.y);
			o_col_pts.push_back(o_max_pt.x);

			o_bin_img.at<uchar>( o_max_pt.y, o_max_pt.x ) = 255;

		}else{
			double d_max_area = 0;
			vector<Point> o_max_contour;
			for( size_t i = 0; i < o_contours.size(); i++ ){
				double d_area = contourArea( o_contours[i] );
				if( d_area >= d_max_area ){
					d_max_area = d_area;
					o_max_contour = o_contours[i];
				}
			}

			o_max_rect = boundingRect( o_max_contour );
			b_flag = true;
		}

		
	}else{
		b_flag = false;
	}
	
	
	//…………………………………………………………………………………………………
#if 0
	FILE *fp1 = fopen("result_1.txt", "w" );

	for( int hei = 0; hei < n_height; hei++ ){
		for( int wid = 0; wid < n_width; wid++ ){
			if( o_bin_img.at<uchar>( hei, wid ) != 0 ){
				fprintf(fp1,"(x,y) = (%d,%d)\t",wid, hei );
				fprintf(fp1, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( hei, wid ), o_g_channel.at<uchar>( hei, wid ), o_b_channel.at<uchar>( hei, wid ) );
			}
		}
	}
	fclose(fp1);
#endif

	int n_x_mean_pt = 0;
	int n_y_mean_pt = 0;

	if( b_flag ){
		n_x_mean_pt = o_max_rect.x + o_max_rect.width / 2;
		n_y_mean_pt = o_max_rect.y + o_max_rect.height / 2;
	}else{
		for( int i = 0; i < o_row_pts.size(); i++ ){
			n_y_mean_pt += o_row_pts[i];
			n_x_mean_pt += o_col_pts[i];
		}

		if( !o_row_pts.empty() && !o_col_pts.empty() ){
			n_x_mean_pt /= o_col_pts.size();
			n_y_mean_pt /= o_row_pts.size();
		}else{
			n_x_mean_pt = 0;
			n_y_mean_pt = 0;
		}
	}
	

	//--------------------在图像上用小的正方形画出激光点--------------------
	Point o_left_top;
	Point o_right_down;
	const int n_offset = 10;
	if( n_x_mean_pt - n_offset >= 0 ){
		o_left_top.x = n_x_mean_pt - n_offset;
	}else{
		o_left_top.x = 0;
	}

	if(n_y_mean_pt - n_offset >= 0){
		o_left_top.y = n_y_mean_pt - n_offset;
	}else{
		o_left_top.y = 0;
	}

	if( n_x_mean_pt + n_offset <= n_width - 1){
		o_right_down.x = n_x_mean_pt + n_offset;
	}else{
		o_right_down.x = n_width - 1;
	}

	if( n_y_mean_pt + n_offset <= n_height - 1){
		o_right_down.y = n_y_mean_pt + n_offset;
	}else{
		o_right_down.y = n_height - 1;
	}
	
	rectangle( o_src_img, o_left_top, o_right_down, CV_RGB(0, 255, 0), 4 );

	//----------------------------------------------------------------------------
	o_src_img.copyTo(om_dst_img);
	m_laser_point_coords = Point( n_x_mean_pt, n_y_mean_pt );

	/*namedWindow("src_img", 0);
	imshow("src_img", o_src_img );
	namedWindow("bin_img", 0 );
	imshow("bin_img", o_bin_img );
	waitKey(0);

	destroyAllWindows();*/

	return 0;
}

/*-----------------------------------------------------------
 *
 * function : 绿色激光点检测算法
 * 
 * parameter: 
 *		Mat o_intput_img  输入的原图像
 *-------------------------------------------------------------
 */
int LaserPointPos::m_green_laser_point_detect( Mat o_input_img )
{
	if( o_input_img.empty() ){
		cout << "the input image of the funtion m_green_laser_point_detect() is empty!" << endl;
		return -1;
	}

	m_laser_point_coords = Point(0,0);

 	Mat o_bin_img(o_input_img.rows, o_input_img.cols, CV_8UC1 );
	o_bin_img = Scalar::all(0);
	//cv::cvtColor( o_input_img, o_input_img, CV_BGR2RGB );
	Mat o_src_img;
	o_input_img.copyTo(o_src_img);
	//namedWindow("src_img1", 0 );
	//imshow("src_img1", o_src_img );
	//waitKey(0);
	//m_roi_detect( o_src_img );

	//blur( o_src_img, o_src_img, Size(5,5) );
	//medianBlur( o_src_img, o_src_img, 3 );
	vector<Mat> o_channels;
	Mat o_r_channel;
	Mat o_g_channel;
	Mat o_b_channel;

	split( o_src_img, o_channels );
	o_b_channel = o_channels.at(0);
	o_g_channel = o_channels.at(1);
	o_r_channel = o_channels.at(2);


	int n_height = o_input_img.rows;
	int n_width  = o_input_img.cols;

	//for( int h = 0; h < n_height; ++h ){
	//	for( int w = 0; w < n_width; ++w ){
	//		o_b_channel.at<uchar>( h, w ) = o_b_channel.at<uchar>( h, w ) /*& om_roi_bin_img.at<uchar>( h, w )*/;
	//		o_g_channel.at<uchar>( h, w ) = o_g_channel.at<uchar>( h, w ) /*& om_roi_bin_img.at<uchar>( h, w )*/;
	//		o_r_channel.at<uchar>( h, w ) = o_r_channel.at<uchar>( h, w ) /*& om_roi_bin_img.at<uchar>( h, w )*/;
	//	}
	//}

	/*uchar *p;
	double d_min_val = 0;
	double d_max_val = 0;
	Point o_min_val_pt;
	Point o_max_val_pt;*/
	//minMaxLoc( o_r_channel, &d_min_val, &d_max_val, &o_min_val_pt, &o_max_val_pt );
	
	int n_max_val = 0;
	int n_min_val = 65535;
	int n_max_sub_val = 0;

	//********************************************************************
	/*
	 * 考虑到绿色激光点照射的地方肯定是 G 值较大的或者是最大的点
	 * 为了滤除噪声和白色像素点，对 R、B 通道进行大小限制
	 * 得到 R 通道中 最大值
	 */
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
			if(  o_g_channel.at<uchar>(row, col) > n_max_val && ( o_r_channel.at<uchar>(row, col) < 175 || o_b_channel.at<uchar>(row, col) < 175 ) ){ //150
					n_max_val = o_g_channel.at<uchar>(row, col);
			}
		}
	}
	//*************************************************************************

	vector<int> o_row_pts;
	vector<int> o_col_pts;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * 检测激光点，即 G 通道在最大值附近，且 G 分别和 R、B 通道的差值大于一定的值
	 */
	
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
#if 0
			if( n_max_val > 240 ){
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && o_g_channel.at<uchar>(row, col) < 70 && o_b_channel.at<uchar>(row, col) < 70  ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}else{
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && ( o_r_channel.at<uchar>(row, col) - o_g_channel.at<uchar>(row, col) > 70 || 
					o_r_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}
#else
			if( o_g_channel.at<uchar>(row, col) >= uchar(n_max_val - 2) && ( o_g_channel.at<uchar>(row, col) - o_r_channel.at<uchar>(row, col) > 70 || 
				o_g_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
					o_bin_img.at<uchar>(row,col) = 255;
					
			}

#endif
			
		}
	}
	
	//FILE *fp;
	//fp = fopen("test.txt", "w" );
#if 1
	//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int n_max_sub_val_GR = 0; // G 分量和 R 分量的最大差值
	int n_max_sub_val_GB = 0; // G 分量和 B 分量的最大差值
	//…………………………………………………………………………………………………
	/*
	 * 寻找候选目标点中， G 分量和 R 分量的最大差值，G 分量和 B 分量的最大差值
	 */
	for( int h = 0; h < n_height; h++ ){
		for( int w = 0; w < n_width; w++ ){
			if( o_bin_img.at<uchar>( h, w ) == 255 ){
				//fprintf(fp,"(x,y) = (%d,%d)\t",w, h );
				//fprintf(fp, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( h, w ), o_g_channel.at<uchar>( h, w ), o_b_channel.at<uchar>( h, w ) );

				if( ( o_g_channel.at<uchar>( h, w ) -  o_r_channel.at<uchar>( h, w ) ) > n_max_sub_val_GR ){
					n_max_sub_val_GR = int( o_g_channel.at<uchar>( h, w ) - o_r_channel.at<uchar>( h, w ) );
				}

				if( ( o_g_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) ) > n_max_sub_val_GB ){
					n_max_sub_val_GB = int(  o_g_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) );
				}

			}else{
				continue;
			}
		}
	}
	//fclose(fp);
	for( int r = 0; r < n_height; r++ ){
		for( int c = 0; c < n_width; c++ ){
			if( o_bin_img.at<uchar>( r, c ) != 0  ){
				if( ( o_g_channel.at<uchar>( r, c ) -  o_b_channel.at<uchar>( r, c ) ) >= (n_max_sub_val_GB - 2 ) ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}

			}else{
				continue;
			}
		}
	}
#else

#endif

	Rect o_max_rect;
	bool b_flag = true;
	if( o_row_pts.empty() || o_col_pts.empty() ){
		for( int r = 0; r < n_height; r++ ){
			for( int c = 0; c < n_width; c++ ){
				
				if(  o_g_channel.at<uchar>( r, c ) >= 195 && o_r_channel.at<uchar>( r, c ) >= 180 && o_b_channel.at<uchar>(r, c) >= 170  ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					o_bin_img.at<uchar>( r, c ) = 255;
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}

				
			}
		}
	
		vector<vector<Point>> o_contours;

		findContours( o_bin_img, o_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

		if( o_contours.empty() ){ //如果没有找到候选激光点区域，就默认投影区域中最亮的像素点就是激光点
			b_flag = false;
			int n_max_sum = 0;
			int n_temp_val = 0;
			Point o_max_pt = Point(0,0);
			for( int r = 0; r < n_height; r++ ){
				for( int c = 0; c < n_width; c++ ){

					n_temp_val = o_r_channel.at<uchar>( r, c ) + o_g_channel.at<uchar>( r, c ) + o_b_channel.at<uchar>(r, c) ;
					if( n_temp_val > n_max_sum ){
						o_max_pt.x = c;
						o_max_pt.y = r;
					}else{
						o_bin_img.at<uchar>( r, c ) = 0;
					}
					/*if(  o_r_channel.at<uchar>( r, c ) >= 190 && o_g_channel.at<uchar>( r, c ) >= 170 && o_b_channel.at<uchar>(r, c) >= 170  ){
						o_row_pts.push_back(r);
						o_col_pts.push_back(c);
						o_bin_img.at<uchar>( r, c ) = 255;
						continue;
					}else{
						o_bin_img.at<uchar>( r, c ) = 0;
					}*/


				}
			}

			o_row_pts.push_back(o_max_pt.y);
			o_col_pts.push_back(o_max_pt.x);

			o_bin_img.at<uchar>( o_max_pt.y, o_max_pt.x ) = 255;

		}else{
			double d_max_area = 0;
			vector<Point> o_max_contour;
			for( size_t i = 0; i < o_contours.size(); i++ ){
				double d_area = contourArea( o_contours[i] );
				if( d_area >= d_max_area ){
					d_max_area = d_area;
					o_max_contour = o_contours[i];
				}
			}

			o_max_rect = boundingRect( o_max_contour );
			b_flag = true;
		}

		
	}else{
		b_flag = false;
	}
	
	
	//…………………………………………………………………………………………………
#if 0
	FILE *fp1 = fopen("result_1.txt", "w" );

	for( int hei = 0; hei < n_height; hei++ ){
		for( int wid = 0; wid < n_width; wid++ ){
			if( o_bin_img.at<uchar>( hei, wid ) != 0 ){
				fprintf(fp1,"(x,y) = (%d,%d)\t",wid, hei );
				fprintf(fp1, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( hei, wid ), o_g_channel.at<uchar>( hei, wid ), o_b_channel.at<uchar>( hei, wid ) );
			}
		}
	}
	fclose(fp1);
#endif

	int n_x_mean_pt = 0;
	int n_y_mean_pt = 0;

	if( b_flag ){
		n_x_mean_pt = o_max_rect.x + o_max_rect.width / 2;
		n_y_mean_pt = o_max_rect.y + o_max_rect.height / 2;
	}else{
		for( int i = 0; i < o_row_pts.size(); i++ ){
			n_y_mean_pt += o_row_pts[i];
			n_x_mean_pt += o_col_pts[i];
		}

		if( !o_row_pts.empty() && !o_col_pts.empty() ){
			n_x_mean_pt /= o_col_pts.size();
			n_y_mean_pt /= o_row_pts.size();
		}else{
			n_x_mean_pt = 0;
			n_y_mean_pt = 0;
		}
	}
	

	//--------------------在图像上用小的正方形画出激光点--------------------
	/*Point o_left_top;
	Point o_right_down;
	const int n_offset = 10;
	if( n_x_mean_pt - n_offset >= 0 ){
		o_left_top.x = n_x_mean_pt - n_offset;
	}else{
		o_left_top.x = 0;
	}

	if(n_y_mean_pt - n_offset >= 0){
		o_left_top.y = n_y_mean_pt - n_offset;
	}else{
		o_left_top.y = 0;
	}

	if( n_x_mean_pt + n_offset <= n_width - 1){
		o_right_down.x = n_x_mean_pt + n_offset;
	}else{
		o_right_down.x = n_width - 1;
	}

	if( n_y_mean_pt + n_offset <= n_height - 1){
		o_right_down.y = n_y_mean_pt + n_offset;
	}else{
		o_right_down.y = n_height - 1;
	}
	
	rectangle( o_src_img, o_left_top, o_right_down, CV_RGB(0, 255, 0), 4 );*/

	//----------------------------------------------------------------------------
	m_draw_laser_rect( o_src_img, Point(n_x_mean_pt,n_y_mean_pt), CV_RGB(128,0,255), 4, 2 );
	o_src_img.copyTo(om_dst_img);
	m_laser_point_coords = Point( n_x_mean_pt, n_y_mean_pt );

	/*namedWindow("src_img", 0);
	imshow("src_img", o_src_img );
	namedWindow("bin_img", 0 );
	imshow("bin_img", o_bin_img );
	waitKey(0);

	destroyAllWindows();*/

	return 0;
}
/*-----------------------------------------------------------
 *
 * function : 绿色激光点检测算法
 * 
 * parameter: 
 *		Mat o_intput_img  输入的原图像
 *-------------------------------------------------------------
 */
int LaserPointPos::m_green_laser_point_detect1( Mat o_input_img )
{
	if( o_input_img.empty() ){
		cout << "the input image of the funtion m_green_laser_point_detect1() is empty!" << endl;
		return -1;
	}

	m_laser_point_coords = Point(0,0);

 	Mat o_bin_img(o_input_img.rows, o_input_img.cols, CV_8UC1 );
	o_bin_img = Scalar::all(0);
	//cv::cvtColor( o_input_img, o_input_img, CV_BGR2RGB );
	Mat o_src_img;
	o_input_img.copyTo(o_src_img);
	//namedWindow("src_img1", 0 );
	//imshow("src_img1", o_src_img );
	//waitKey(0);
	//m_roi_detect( o_src_img );

	//blur( o_src_img, o_src_img, Size(5,5) );
	//medianBlur( o_src_img, o_src_img, 3 );
	vector<Mat> o_channels;
	Mat o_r_channel;
	Mat o_g_channel;
	Mat o_b_channel;

	split( o_src_img, o_channels );
	o_b_channel = o_channels.at(0);
	o_g_channel = o_channels.at(1);
	o_r_channel = o_channels.at(2);

	int n_height = o_input_img.rows;
	int n_width  = o_input_img.cols;

	for( int h = 0; h < n_height; ++h ){
		for( int w = 0; w < n_width; ++w ){
			o_b_channel.at<uchar>( h, w ) = o_b_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
			o_g_channel.at<uchar>( h, w ) = o_g_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
			o_r_channel.at<uchar>( h, w ) = o_r_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
		}
	}

	/*uchar *p;
	double d_min_val = 0;
	double d_max_val = 0;
	Point o_min_val_pt;
	Point o_max_val_pt;*/
	//minMaxLoc( o_r_channel, &d_min_val, &d_max_val, &o_min_val_pt, &o_max_val_pt );
	
	int n_max_val = 0;
	int n_min_val = 65535;
	int n_max_sub_val = 0;

	//********************************************************************
	/*
	 * 考虑到绿色激光点照射的地方肯定是 G 值较大的或者是最大的点
	 * 为了滤除噪声和白色像素点，对 R、B 通道进行大小限制
	 * 得到 R 通道中 最大值
	 */
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
			if(  o_g_channel.at<uchar>(row, col) > n_max_val && ( o_r_channel.at<uchar>(row, col) < 175 || o_b_channel.at<uchar>(row, col) < 175 ) ){ //150
					n_max_val = o_g_channel.at<uchar>(row, col);
			}
		}
	}
	//*************************************************************************

	vector<int> o_row_pts;
	vector<int> o_col_pts;
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * 检测激光点，即 G 通道在最大值附近，且 G 分别和 R、B 通道的差值大于一定的值
	 */
	
	for( int row = 0; row < n_height; row ++ ){
		for( int col = 0; col < n_width; col ++ ){
#if 0
			if( n_max_val > 240 ){
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && o_g_channel.at<uchar>(row, col) < 70 && o_b_channel.at<uchar>(row, col) < 70  ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}else{
				if( o_r_channel.at<uchar>(row, col) >= uchar(n_max_val - 5) && ( o_r_channel.at<uchar>(row, col) - o_g_channel.at<uchar>(row, col) > 70 || 
					o_r_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
						o_bin_img.at<uchar>(row,col) = 255;
						o_row_pts.push_back(row);
						o_col_pts.push_back(col);
				}
			}
#else
			if( o_g_channel.at<uchar>(row, col) >= uchar(n_max_val - 2) && ( o_g_channel.at<uchar>(row, col) - o_r_channel.at<uchar>(row, col) > 70 || 
				o_g_channel.at<uchar>(row, col) - o_b_channel.at<uchar>(row, col) > 70 ) ){
					o_bin_img.at<uchar>(row,col) = 255;
					
			}

#endif
			
		}
	}
	
	//FILE *fp;
	//fp = fopen("test.txt", "w" );
#if 1
	//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	int n_max_sub_val_GR = 0; // G 分量和 R 分量的最大差值
	int n_max_sub_val_GB = 0; // G 分量和 B 分量的最大差值
	//…………………………………………………………………………………………………
	/*
	 * 寻找候选目标点中， G 分量和 R 分量的最大差值，G 分量和 B 分量的最大差值
	 */
	for( int h = 0; h < n_height; h++ ){
		for( int w = 0; w < n_width; w++ ){
			if( o_bin_img.at<uchar>( h, w ) == 255 ){
				//fprintf(fp,"(x,y) = (%d,%d)\t",w, h );
				//fprintf(fp, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( h, w ), o_g_channel.at<uchar>( h, w ), o_b_channel.at<uchar>( h, w ) );
				if( ( o_g_channel.at<uchar>( h, w ) -  o_r_channel.at<uchar>( h, w ) ) > n_max_sub_val_GR ){
					n_max_sub_val_GR = int( o_g_channel.at<uchar>( h, w ) - o_r_channel.at<uchar>( h, w ) );
				}

				if( ( o_g_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) ) > n_max_sub_val_GB ){
					n_max_sub_val_GB = int( o_g_channel.at<uchar>( h, w ) - o_b_channel.at<uchar>( h, w ) );
				}

			}else{
				continue;
			}
		}
	}
	//fclose(fp);
	for( int r = 0; r < n_height; r++ ){
		for( int c = 0; c < n_width; c++ ){
			if( o_bin_img.at<uchar>( r, c ) != 0  ){
				if( ( o_g_channel.at<uchar>( r, c ) -  o_b_channel.at<uchar>( r, c ) ) >= (n_max_sub_val_GB - 2 ) ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}

			}else{
				continue;
			}
		}
	}
#else

#endif

	Rect o_max_rect;
	bool b_flag = true;
	if( o_row_pts.empty() || o_col_pts.empty() ){
		for( int r = 0; r < n_height; r++ ){
			for( int c = 0; c < n_width; c++ ){
				
				if( o_g_channel.at<uchar>( r, c ) >= 195 && o_r_channel.at<uchar>( r, c ) >= 180 && o_b_channel.at<uchar>(r, c) >= 170 ){
					o_row_pts.push_back(r);
					o_col_pts.push_back(c);
					o_bin_img.at<uchar>( r, c ) = 255;
					continue;
				}else{
					o_bin_img.at<uchar>( r, c ) = 0;
				}
				
			}
		}
	
		vector<vector<Point>> o_contours;

		findContours( o_bin_img, o_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

		if( o_contours.empty() ){ //如果没有找到候选激光点区域，就默认投影区域中最亮的像素点就是激光点
			b_flag = false;
			int n_max_sum = 0;
			int n_temp_val = 0;
			Point o_max_pt = Point(0,0);
			for( int r = 0; r < n_height; r++ ){
				for( int c = 0; c < n_width; c++ ){

					n_temp_val = o_r_channel.at<uchar>( r, c ) + o_g_channel.at<uchar>( r, c ) + o_b_channel.at<uchar>(r, c) ;
					if( n_temp_val > n_max_sum ){
						o_max_pt.x = c;
						o_max_pt.y = r;
					}else{
						o_bin_img.at<uchar>( r, c ) = 0;
					}
					/*if(  o_r_channel.at<uchar>( r, c ) >= 190 && o_g_channel.at<uchar>( r, c ) >= 170 && o_b_channel.at<uchar>(r, c) >= 170  ){
						o_row_pts.push_back(r);
						o_col_pts.push_back(c);
						o_bin_img.at<uchar>( r, c ) = 255;
						continue;
					}else{
						o_bin_img.at<uchar>( r, c ) = 0;
					}*/


				}
			}

			o_row_pts.push_back(o_max_pt.y);
			o_col_pts.push_back(o_max_pt.x);

			o_bin_img.at<uchar>( o_max_pt.y, o_max_pt.x ) = 255;

		}else{
			double d_max_area = 0;
			vector<Point> o_max_contour;
			for( size_t i = 0; i < o_contours.size(); i++ ){
				double d_area = contourArea( o_contours[i] );
				if( d_area >= d_max_area ){
					d_max_area = d_area;
					o_max_contour = o_contours[i];
				}
			}

			o_max_rect = boundingRect( o_max_contour );
			b_flag = true;
		}

		
	}else{
		b_flag = false;
	}
	
	
	//…………………………………………………………………………………………………
#if 0
	FILE *fp1 = fopen("result_1.txt", "w" );

	for( int hei = 0; hei < n_height; hei++ ){
		for( int wid = 0; wid < n_width; wid++ ){
			if( o_bin_img.at<uchar>( hei, wid ) != 0 ){
				fprintf(fp1,"(x,y) = (%d,%d)\t",wid, hei );
				fprintf(fp1, "R:%d,G:%d,B:%d\n", o_r_channel.at<uchar>( hei, wid ), o_g_channel.at<uchar>( hei, wid ), o_b_channel.at<uchar>( hei, wid ) );
			}
		}
	}
	fclose(fp1);
#endif

	int n_x_mean_pt = 0;
	int n_y_mean_pt = 0;

	if( b_flag ){
		n_x_mean_pt = o_max_rect.x + o_max_rect.width / 2;
		n_y_mean_pt = o_max_rect.y + o_max_rect.height / 2;
	}else{
		for( int i = 0; i < o_row_pts.size(); i++ ){
			n_y_mean_pt += o_row_pts[i];
			n_x_mean_pt += o_col_pts[i];
		}

		if( !o_row_pts.empty() && !o_col_pts.empty() ){
			n_x_mean_pt /= o_col_pts.size();
			n_y_mean_pt /= o_row_pts.size();
		}else{
			n_x_mean_pt = 0;
			n_y_mean_pt = 0;
		}
	}
	

	//--------------------在图像上用小的正方形画出激光点--------------------
	/*Point o_left_top;
	Point o_right_down;
	const int n_offset = 10;
	if( n_x_mean_pt - n_offset >= 0 ){
		o_left_top.x = n_x_mean_pt - n_offset;
	}else{
		o_left_top.x = 0;
	}

	if(n_y_mean_pt - n_offset >= 0){
		o_left_top.y = n_y_mean_pt - n_offset;
	}else{
		o_left_top.y = 0;
	}

	if( n_x_mean_pt + n_offset <= n_width - 1){
		o_right_down.x = n_x_mean_pt + n_offset;
	}else{
		o_right_down.x = n_width - 1;
	}

	if( n_y_mean_pt + n_offset <= n_height - 1){
		o_right_down.y = n_y_mean_pt + n_offset;
	}else{
		o_right_down.y = n_height - 1;
	}
	
	rectangle( o_src_img, o_left_top, o_right_down, CV_RGB(0, 255, 0), 4 );*/

	//----------------------------------------------------------------------------
	m_draw_laser_rect( o_src_img, Point(n_x_mean_pt,n_y_mean_pt), CV_RGB(128,0,255), 4, 2 );
	o_src_img.copyTo(om_dst_img);
	m_laser_point_coords = Point( n_x_mean_pt, n_y_mean_pt );

	/*namedWindow("src_img", 0);
	imshow("src_img", o_src_img );
	namedWindow("bin_img", 0 );
	imshow("bin_img", o_bin_img );
	waitKey(0);

	destroyAllWindows();*/

	return 0;
}

/*-----------------------------------------------------------
 *
 * function : 绿色激光点检测算法,利用HSV颜色空间信息
 * 
 * parameter: 
 *		Mat o_intput_img  输入的原图像
 *-------------------------------------------------------------
 */
int LaserPointPos::m_green_laser_point_detect_by_hsv( Mat o_input_img )
{
	if( o_input_img.empty() || o_input_img.rows <= 0 || o_input_img.cols <= 0 ){
		cout << "the input image of the funtion m_green_laser_point_detect_by_hsv() is empty!" << endl;
		return -1;
	}

	int n_height = o_input_img.rows;
	int n_width  = o_input_img.cols;

	uchar *p_buf_b;
	uchar *p_buf_g;
	uchar *p_buf_r;

	float *p_buf_h = new float[n_height * n_width];
	float *p_buf_s = new float[n_height * n_width];
	float *p_buf_v = new float[n_height * n_width];

	m_laser_point_coords = Point(0,0);

	//Mat o_bin_img(o_input_img.rows, o_input_img.cols, CV_8UC1 );
	//o_bin_img = Scalar::all(0);
	//cv::cvtColor( o_input_img, o_input_img, CV_BGR2RGB );
	Mat o_src_img;
	o_input_img.copyTo(o_src_img);
	//namedWindow("src_img1", 0 );
	//imshow("src_img1", o_src_img );
	//waitKey(0);
	
	vector<Mat> o_channels;
	Mat o_r_channel;
	Mat o_g_channel;
	Mat o_b_channel;

	split( o_src_img, o_channels );
	o_b_channel = o_channels.at(0);
	o_g_channel = o_channels.at(1);
	o_r_channel = o_channels.at(2);

	/*for( int h = 0; h < n_height; ++h ){
		for( int w = 0; w < n_width; ++w ){
			o_b_channel.at<uchar>( h, w ) = o_b_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
			o_g_channel.at<uchar>( h, w ) = o_g_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
			o_r_channel.at<uchar>( h, w ) = o_r_channel.at<uchar>( h, w ) & om_roi_bin_img.at<uchar>( h, w );
		}
	}*/

	p_buf_b = o_b_channel.data;
	p_buf_g = o_g_channel.data;
	p_buf_r = o_r_channel.data;

	for( int i = 0; i < n_height * n_width; i++ ){
		m_rgb_to_hsv( p_buf_r[i], p_buf_g[i], p_buf_b[i], p_buf_h[i], p_buf_s[i], p_buf_v[i] );
	}

	vector<int> o_row_pts;
	vector<int> o_col_pts;

	for( int h = 0; h < n_height; h++ ){
		for( int w = 0; w < n_width; w++ ){ //version0.1:(180,90,0.35,180);ver:(160,110,0.35,180)
			if( p_buf_h[h * n_width + w] <= 165 && p_buf_h[h * n_width + w] >= 80 && p_buf_s[h * n_width +w] >= 0.35 && p_buf_v[h * n_width + w] >= 180 ){
				//draw_rect( o_result_img, cv::Point(w,h), CV_RGB(128,0,255), 4, 4 );
				o_row_pts.push_back( h );
				o_col_pts.push_back( w );
			}
		}
	}

	int n_x_mean_pt = 0;
	int n_y_mean_pt = 0;

	for( int i = 0; i < o_row_pts.size(); i++ ){
		n_y_mean_pt += o_row_pts[i];
		n_x_mean_pt += o_col_pts[i];
	}

	if( !o_row_pts.empty() && !o_col_pts.empty() ){
		n_x_mean_pt /= o_col_pts.size();
		n_y_mean_pt /= o_row_pts.size();
	}else{
		n_x_mean_pt = 0;
		n_y_mean_pt = 0;
	}

	if( n_x_mean_pt > 0 && n_y_mean_pt > 0 ){
		//m_draw_laser_rect( o_src_img, Point(n_x_mean_pt,n_y_mean_pt), CV_RGB(128,0,255), 4, 2 );
		m_draw_rect( o_src_img, Point(n_x_mean_pt,n_y_mean_pt), CV_RGB(128,0,255), 2, 2 );
	}else{

	}
	/*namedWindow("test1",1);
	imshow("test1",o_src_img);
	waitKey();*/
	o_src_img.copyTo(om_dst_img);
	m_laser_point_coords = Point( n_x_mean_pt, n_y_mean_pt );

	delete []p_buf_h;
	delete []p_buf_s;
	delete []p_buf_v;

	if( p_buf_b != NULL ){
		p_buf_b = NULL;
	}

	if( p_buf_g != NULL ){
		p_buf_g = NULL;
	}

	if( p_buf_r != NULL ){
		p_buf_r = NULL;
	}

	return 0;
}

int LaserPointPos::m_rgb_to_hsv( uchar uc_r, uchar uc_g, uchar uc_b, float &f_h, float &f_s, float &f_v )
{
	uchar nc_min_val, nc_max_val;
	float f_delta;

	nc_min_val = Min( uc_r, uc_g, uc_b );
	nc_max_val = Max( uc_r, uc_g, uc_b );
	f_v = nc_max_val; // f_v
	f_delta = nc_max_val - nc_min_val;

	if( nc_max_val != 0 )
	{
		f_s = f_delta / nc_max_val; // f_s
	}
	else
	{
		// uc_r = uc_g = uc_b = 0 // f_s = 0, f_v is undefined
		f_s = 0;
		f_h = -1;
		return -1;
	}

	if( uc_r == nc_max_val )
	{
		f_h = ( uc_g - uc_b ) / f_delta; // between yellow & magenta
	}
	else if( uc_g == nc_max_val )
	{
		f_h = 2 + ( uc_b - uc_r ) / f_delta; // between cyan & yellow
	}
	else
	{
		f_h = 4 + ( uc_r - uc_g ) / f_delta; // between magenta & cyan
	}

	f_h *= 60; // degrees

	if( f_h < 0 )
	{
		f_h += 360;
	}

	return 0;
}
/*-----------------------------------------------------------------------------
 *
 * function : 在图像中用矩形画出激光点，方便观察
 * 
 * parameter: 
 *		1、IplImage *o_intput_img : 输入要画的图像
 *		2、Point     laser_pos_pt : 激光点在图像中的坐标
 *		3、int       offset       : 矩形的定点相对于激光点的偏移量
 *      4、int       thickness    : 矩形的线宽
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_draw_laser_rect( Mat &input_img, Point2f laser_pos_pt, const cv::Scalar &color, int offset, int thickness )
{
	if( input_img.empty() ){
		cout << "the input image of the function m_draw_laser_rect() is empty!" << endl;
		return -1;
	}
	int n_wid = input_img.cols;
	int n_hei = input_img.rows;
	Point o_left_top;
	Point o_right_down;
	int n_offset = offset; 
	Point o_temp_pt;
	o_temp_pt.x = int( laser_pos_pt.x * om_roi_img.cols );
	o_temp_pt.y = int( laser_pos_pt.y * om_roi_img.rows );
	if( o_temp_pt.x - n_offset >= 0 ){
		o_left_top.x = o_temp_pt.x - n_offset;
	}else{
		o_left_top.x = 0;
	}

	if( o_temp_pt.y - n_offset >= 0 ){
		o_left_top.y = o_temp_pt.y - n_offset;
	}else{
		o_left_top.y = 0;
	}

	if( o_temp_pt.x + n_offset <= n_wid - 1 ){
		o_right_down.x = o_temp_pt.x + n_offset;
	}else{
		o_right_down.x = n_wid - 1;
	}

	if( o_temp_pt.y + n_offset <= n_wid - 1 ){
		o_right_down.y = o_temp_pt.y + n_offset;
	}else{
		o_right_down.y = n_hei - 1;
	}

	rectangle( input_img, o_left_top, o_right_down, color, thickness );

	return 0;

}
/*-----------------------------------------------------------------------------
 *
 * function : 在图像中用矩形画出激光点，方便观察
 * 
 * parameter: 
 *		1、Mat &intput_img : 输入要画的图像
 *		2、Point     pt : 激光点在图像中的坐标
 *		3、int       offset       : 矩形的定点相对于激光点的偏移量
 *      4、int       thickness    : 矩形的线宽
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_draw_rect( cv::Mat &input_img, cv::Point pt, CvScalar &color, int offset, int thickness )
{
	//--------------------在图像上用小的正方形画出激光点--------------------
	int n_hei = input_img.rows;
	int n_wid = input_img.cols;

	cv::Point o_left_top;
	cv::Point o_right_down;
	int n_offset = offset; //在激光点出画正方形，它的边长的一半
	if( pt.x - n_offset >= 0 ){
		o_left_top.x = pt.x - n_offset;
	}else{
		o_left_top.x = 0;
	}

	if( pt.y - n_offset >= 0 ){
		o_left_top.y = pt.y - n_offset;
	}else{
		o_left_top.y = 0;
	}

	if( pt.x + n_offset <= n_wid - 1 ){
		o_right_down.x = pt.x + n_offset;
	}else{
		o_right_down.x = n_wid - 1;
	}

	if( pt.y + n_offset <= n_hei - 1 ){
		o_right_down.y = pt.y + n_offset;
	}else{
		o_right_down.y = n_hei - 1;
	}

	cv::rectangle( input_img, o_left_top, o_right_down, color, thickness );

	return 0;
}

/*-----------------------------------------------------------------------------
 *
 * function : 对图像中投影区域即ROI区域进行激光点检测，
 *             并得到校正后roi区域和激光点在其中的坐标        
 * 
 * parameter: 
 *		1、IplImage *o_intput_img : 输入彩色图像
 *      2、laser_type type        : 激光类型，即对何种颜色的激光进行检测
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_roi_detect( Mat o_input_img, laser_type type )
{
	/*qint64 i64_current_time = 0;
	qint64 i64_last_time = 0;
	vector<double> o_spend_time_vec;*/
	if( o_input_img.empty() ){
		cout << "the input image of the funtion m_roi_detect() is empty!" << endl;
		return -1;
	}

	m_laser_point_coords = Point(0,0);
	m_adjust_laser_pt = Point2f(0,0);
	m_norm_img_laser_pt = Point(0,0);

	Mat o_temp_img = o_input_img.clone();
	Mat o_temp_img1;
	o_input_img.copyTo( o_temp_img1 );
	//resize( o_temp_img, o_temp_img, Size( o_temp_img.cols/2, o_temp_img.rows/2), 0, 0, INTER_AREA );
	Mat gray_img( o_temp_img.rows, o_temp_img.cols, CV_8UC1 );
	cvtColor( o_temp_img, o_temp_img, COLOR_BGR2HSV );
	cvtColor( o_temp_img, gray_img, COLOR_BGR2GRAY );

	vector<Mat> o_hsv_channels_vec;
	//Mat o_h_img;  
	//Mat o_s_img;  
	Mat o_v_img;   
	// 把一个3通道图像转换成3个单通道图像  
	split(o_temp_img, o_hsv_channels_vec );//分离色彩通道  
	//o_h_img = o_hsv_channels_vec.at(0); 
	//o_s_img = o_hsv_channels_vec.at(1);  
	o_v_img = o_hsv_channels_vec.at(2);  

	threshold( o_v_img, o_v_img,  0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU );
	Mat o_img_col( gray_img.rows, gray_img.cols, CV_8UC3 );
	//dilate( o_r_img, o_r_img, Mat());
	dilate(o_v_img, o_v_img,Mat(4, 1, CV_8U, cvScalar(1)), Point( 0, 0), 1 );

	//i64_last_time = QDateTime::currentMSecsSinceEpoch();
	m_contours_filter( &IplImage(o_v_img), &IplImage(o_img_col), gray_img.rows * gray_img.cols * 0.2 );
	//判断是否检测到有ROI区域
	if( o_roi_rect_size.width < 0 || o_roi_rect_size.height < 0 || o_min_bounding_rect.height < 0 || o_min_bounding_rect.width < 0 ){
		m_adjust_laser_pt = Point2f(-1.0,-1.0);
		return -1;
	}
	/*i64_current_time = QDateTime::currentMSecsSinceEpoch();
	mi_spend_time = i64_current_time - i64_last_time;*/
	//qDebug("%f\n", mi_spend_time );
	//o_spend_time_vec.push_back( double(mi_spend_time) );
	//Mat o_color_dst( gray_img.rows, gray_img.cols, CV_8UC3 );
	Mat o_gray_img;
	Rect o_temp_rect( o_min_bounding_rect );
	cvtColor( o_img_col, o_gray_img, CV_BGR2GRAY );
	Mat o_rect_img;
	o_temp_img1(o_temp_rect).copyTo( o_rect_img );
	o_temp_img1(o_temp_rect).copyTo( om_roi_img );
	o_gray_img( o_temp_rect ).copyTo( om_roi_bin_img );

	m_roi_point_detect( om_roi_bin_img );

	if( GREEN_LASER == type ){
		//m_green_laser_point_detect1( o_rect_img );
		m_green_laser_point_detect_by_hsv( o_rect_img );
	}else{
		m_laser_point_detect1( o_rect_img );
	}

	Mat o_laser_pt_img;
	cvtColor( om_roi_bin_img, o_laser_pt_img, CV_GRAY2BGR );
	cvSet2D( &IplImage(o_laser_pt_img), m_laser_point_coords.y, m_laser_point_coords.x, CV_RGB(255,0,0) );

	om_adjust_roi_img = m_roi_img_adjust( o_rect_img );//2015.10.09
	//om_adjust_roi_bin_img = m_roi_img_adjust( o_laser_pt_img );

	int n_hei = om_adjust_roi_bin_img.rows;
	int n_wid = om_adjust_roi_bin_img.cols;
	IplImage *p_temp_img = &IplImage( om_adjust_roi_bin_img );
	uchar *data = (uchar*)p_temp_img->imageData;
	int n_step = p_temp_img->widthStep;
	int n_channels = p_temp_img->nChannels;
	//-------------找出图像中三个通道像素值不相等的像素坐标--------------------------
	/*for( int h = 0; h < n_hei; h++ ){
		for( int w = 0; w < n_wid; w++ ){
			if( (uchar)data[h * n_step + w * n_channels + 0] != (uchar)data[ h * n_step + w * n_channels + 1] || 
				(uchar)data[h * n_step + w * n_channels + 0] != (uchar)data[ h * n_step + w * n_channels + 2] || 
				(uchar)data[h * n_step + w * n_channels + 1] != (uchar)data[ h * n_step + w * n_channels + 2]){
				m_adjust_laser_pt = Point( w, h );
			}
		}
	}*/
	//--------------------------------------------------------------------------------

	m_draw_laser_rect( om_adjust_roi_img, m_adjust_laser_pt, CV_RGB(128,0,255), 8, 4 );

	/*if( m_laser_point_coords.x >= 300 && m_laser_point_coords.x <= 470 &&
		m_laser_point_coords.y >= 130 && m_laser_point_coords.y <= 310 ) {
			count++;
	}*/

	if( m_adjust_laser_pt.x <= 0 || m_adjust_laser_pt.x >= 1.0 || m_adjust_laser_pt.y <= 0 || m_adjust_laser_pt.y >= 1.0 ) {
			m_adjust_laser_pt.x = -1.0;
			m_adjust_laser_pt.y = -1.0;
	}

	/*if( m_adjust_laser_pt.y <= 0 || m_adjust_laser_pt.y >= 1.0) {
		m_adjust_laser_pt.y = -1.0;
	}*/

	int n_temp_x = int( n_norm_width * m_adjust_laser_pt.x / n_wid );
	int n_temp_y = int( n_norm_height * m_adjust_laser_pt.y / n_hei );
	m_norm_img_laser_pt = Point(n_temp_x, n_temp_y);

	om_adjust_roi_img.copyTo( om_norm_roi_img );
	resize( om_norm_roi_img, om_norm_roi_img, Size(n_norm_width,n_norm_height), 0, 0, INTER_AREA );

	
	/*namedWindow("o_rect_img", 0 );
	namedWindow("v_img",0 );
	namedWindow("roi_bin_img", 0 );
	namedWindow("adjust_roi_img", 0 );
	namedWindow("adjust_roi_bin_img", 0 );
	namedWindow("dst_img", 0 );
	namedWindow("laser_pt_img", 0 );

	imshow("laser_pt_img", o_laser_pt_img );
	imshow("dst_img", om_dst_img );
	imshow("adjust_roi_bin_img", om_adjust_roi_bin_img );
	imshow("adjust_roi_img", om_adjust_roi_img );
	imshow("roi_bin_img", om_roi_bin_img );
	imshow("o_rect_img", o_rect_img);
	imshow("v_img", o_img_col );

	waitKey(); 

	destroyAllWindows();*/
	if( !o_hsv_channels_vec.empty() ){
		o_hsv_channels_vec.clear();
	}

	return count;
}

/*-----------------------------------------------------------------------------
 *
 * function : 填充内部轮廓
 * 
 * parameter: 
 *		1、IplImage *o_intput_img  输入的二值图像
 *		2、IplImage *p_color_img   填充后的彩色图像
 *		3、double   d_area_thre    面积阈值，轮廓的面积根据此阈值进行填充
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_fill_inter_contours( IplImage *p_bin_img, IplImage *p_color_img, double d_area_thre )
{
	if( NULL == p_bin_img->imageData ){
		cout << "the bin image of the funtion m_fill_inter_contours() is empty!" << endl;
		return -1;
	}

	double dConArea;     
	CvSeq *pContour = NULL;     
	CvSeq *pConInner = NULL;     
	CvMemStorage *pStorage = NULL;
	IplImage* img_8uc3 = cvCreateImage(cvGetSize(p_bin_img), IPL_DEPTH_8U, 3);
	// 执行条件 

	if(p_bin_img)     
	{     
		cvNamedWindow("ColorImage",1);
		cvNamedWindow("BinaryImage",1);
		// 查找所有轮廓     
		pStorage = cvCreateMemStorage(0);     
		cvFindContours( p_bin_img, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );     
		// 填充所有轮廓     
		cvDrawContours( p_bin_img, pContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0));    
		// 外轮廓循环     
		int wai = 0;    
		int nei = 0;    
		for (; pContour != NULL; pContour = pContour->h_next)     
		{     
			wai++; 
			dConArea = fabs(cvContourArea(pContour, CV_WHOLE_SEQ));    
			printf("%d_外轮廓面积:%f\n",wai,dConArea);  
					
			if (dConArea < d_area_thre)   //
			{   
				cvCvtColor(p_bin_img,img_8uc3,CV_GRAY2BGR);
				cvDrawContours(img_8uc3, pContour, CV_RGB(255, 255, 0), CV_RGB(255, 0, 0), 0, CV_FILLED, 8 );  
			} 
			
			cvCopy(img_8uc3, p_color_img);
			//cvNamedWindow("ColorImage",0);
			//cvShowImage("ColorImage",img_8uc3);
			//cvWaitKey();
			// 内轮廓循环     
			for (pConInner = pContour->v_next; pConInner != NULL; pConInner = pConInner->h_next)     
			{     
				nei++;    
				// 内轮廓面积     
				dConArea = fabs(cvContourArea(pConInner, CV_WHOLE_SEQ));    
				printf("%d_内轮廓面积:%f\n",nei,dConArea);  

				if (dConArea <= d_area_thre)   
				{   
					cvDrawContours(p_bin_img, pConInner, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, CV_FILLED, 8, cvPoint(0, 0));  
				} 
				
			}   

			cvShowImage("ColorImage",img_8uc3);
			cvShowImage("BinaryImage",p_bin_img);
			printf("wai = %d, nei = %d", wai, nei);

			cvWaitKey();
			//CvRect rect = cvBoundingRect(pContour,0); 
			//cvCvtColor(p_bin_img,img_8uc3,CV_GRAY2BGR);
			//cvRectangle(p_bin_img, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),CV_RGB(255,0, 0), 3, 8, 0);  
		}   



		//cvCopy(img_8uc3, Image_8U3);
		cvReleaseMemStorage(&pStorage);
		cvReleaseImage(&img_8uc3);
		cvDestroyAllWindows();
		pStorage = NULL;     
	}else{
		return -1;
	} 

	return 0;
}

/*-----------------------------------------------------------------------------
 *
 * function : 对候选目标轮廓进行筛选，过滤
 * 
 * parameter: 
 *		1、IplImage *o_intput_img  输入的二值图像
 *      2、IplImage *img_8uc3      彩色图像
 *		3、double   d_area_thre    面积阈值，轮廓的面积根据此阈值进行删除
 * 
 * return   :
 *		int 
 *		
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_contours_filter( IplImage *p_bin_img, IplImage *img_8uc3, double d_area_thre  )
{
	if( NULL == p_bin_img->imageData ){
		cout << "the bin image of the funtion m_contours_filter() is empty!" << endl;
		return -1;
	}
	//是否显示调试图像， true : 显示
	bool b_display_img = false;

	o_min_bounding_rect.height = -1;
	o_min_bounding_rect.width  = -1;
	o_min_bounding_rect.x = -1;
	o_min_bounding_rect.y = -1;

	o_roi_rect_size.height = -1;
	o_roi_rect_size.width  = -1;

	o_roi_rect_center.x = -1;
	o_roi_rect_center.y = -1;

	double dConArea;     
	CvSeq *pContour = NULL;     
	CvSeq *pConInner = NULL; 
	CvMemStorage *pStorage = NULL; 
	//IplImage* img_8uc3 = cvCreateImage(cvGetSize(p_bin_img), IPL_DEPTH_8U, 3);

	if( b_display_img ){
		cvNamedWindow("ColorImage",1);
		cvNamedWindow("BinaryImage",1);
	}
	
    
	if(p_bin_img)     
	{     
		// 查找所有轮廓     
		pStorage = cvCreateMemStorage(0);     
		cvFindContours(p_bin_img, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); 
		cvCvtColor( p_bin_img, img_8uc3, CV_GRAY2BGR );
		// 填充所有轮廓     
		cvDrawContours(p_bin_img, pContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0)); 
		cvDrawContours(img_8uc3, pContour, CV_RGB(0, 0, 0), CV_RGB(0, 0, 0), 2, CV_FILLED, 8, cvPoint(0, 0)); 
		int n_area_max = int(d_area_thre);
		for (; pContour != NULL; pContour = pContour->h_next)     
		{   

			CvBox2D rect = cvMinAreaRect2(pContour,pStorage ); 
			//找出面积最大的区域
			if( rect.size.height * rect.size.width > n_area_max ){ 
				cvDrawContours(img_8uc3, pContour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 0, CV_FILLED, 8 ); 
				n_area_max = rect.size.height * rect.size.width ;
				o_min_bounding_rect = cvBoundingRect( pContour, 1 );
				//得到roi区域的最小外接矩形的中心点坐标和尺寸大小
				o_roi_rect_center.x = rect.center.x;
				o_roi_rect_center.y = rect.center.y;
				o_roi_rect_size.height = rect.size.height;
				o_roi_rect_size.width  = rect.size.width;
			}else{
				cvDrawContours(img_8uc3, pContour, CV_RGB(0, 0, 0), CV_RGB(0, 0, 0), 0, CV_FILLED, 8 );

			}

			if( b_display_img ){
				cvShowImage("ColorImage",img_8uc3);
				cvShowImage("BinaryImage",p_bin_img);

				cvWaitKey();
			}
			 
			   
		}   

		cvReleaseMemStorage(&pStorage);

		if( b_display_img ){
			cvDestroyAllWindows();
		}
		
		pStorage = NULL; 
	}  

	return -1;
}

/*-----------------------------------------------------------------------------
 *
 * function : 对二值图像进行多边形拟合
 * 
 * parameter: 
 *		1、IplImage *o_intput_img  输入的二值图像
 * 
 * return   :
 *		IplImage *  拟合后只显示多边形的图像
 *		
 *-----------------------------------------------------------------------------
 */
IplImage *LaserPointPos::m_polygonal_fitting( IplImage *p_bin_img )
{
	if( NULL == p_bin_img->imageData ){
		cout << "the bin image of the funtion m_polygonal_fitting() is empty!" << endl;
		exit(-1);
	}

	IplImage *p_bin_img_copy = cvCreateImage(cvGetSize(p_bin_img), IPL_DEPTH_8U, 1);  
	//IplImage *p_gray_img = NULL;  
	IplImage *p_dst_img = cvCreateImage(cvGetSize(p_bin_img), IPL_DEPTH_8U, 3);;  

	CvMemStorage *p_storage = cvCreateMemStorage (0);  
	CvMemStorage *p_storage1 = cvCreateMemStorage (0);  
	CvSeq *p_contour = 0;  
	CvSeq *p_cont;  
	CvSeq *p_mcont; 

	int n_count = 0;

	cvCopy(p_bin_img,p_bin_img_copy); 
	cvCvtColor( p_bin_img_copy, p_dst_img, CV_GRAY2BGR );
	cvFindContours (p_bin_img_copy, p_storage, &p_contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);  

	if( p_contour ){  
		CvTreeNodeIterator o_iterator;  
		cvInitTreeNodeIterator( &o_iterator, p_contour, 1 );  
		while (0 != ( p_cont = (CvSeq*)cvNextTreeNode( &o_iterator ) ) ){  
			n_count++;
			p_mcont = cvApproxPoly (p_cont, sizeof(CvContour), p_storage1, CV_POLY_APPROX_DP, cvContourPerimeter(p_cont)*0.02,0);  
			cvDrawContours(p_dst_img, p_mcont , CV_RGB(255,0,0), CV_RGB(0,0,100), 1, 2, 8, cvPoint(0,0) );  

		}  
	}  
	//printf("---------------\ncount : %d---------------\n",n_count);
	//cvNamedWindow("Contour", 1);
	//cvNamedWindow("srcImage", 1);
	//cvShowImage("Contour", p_dst_img);
	//cvShowImage("srcImage",p_src_img);

	//cvWaitKey (0);  

	cvReleaseMemStorage(&p_storage);  
	cvReleaseImage(&p_bin_img_copy);
	//destroyAllWindows();
	return p_dst_img; 
}

/*-----------------------------------------------------------------------------
 *
 * function : 检测图像中的直线
 * 
 * parameter: 
 *		1、Mat o_intput_img  输入的图像
 * 	
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_line_detect( Mat o_input_img )
{
	if( NULL == o_input_img.data ){
		cout << "the input image of the funtion m_line_detect() is empty!" << endl;
		return -1;
	}
	Mat o_gray_img;
	Mat o_color_dst;
	Mat o_edge_img;
	if( 3 == o_input_img.channels() ){
		cvtColor( o_input_img, o_gray_img, CV_BGR2GRAY );
	}else{
		o_input_img.copyTo( o_gray_img );
	}
	Canny( o_gray_img, o_edge_img, 50, 150, 3 );
	cvtColor( o_gray_img, o_color_dst, CV_GRAY2BGR );
	//-----------------------直线检测---------------------
#if 0
		vector<Vec2f> o_lines;
	HoughLines( o_edge_img, o_lines, 1, CV_PI / 180, 200 );
	for( size_t i = 0; i < o_lines.size(); i++ ){
		float f_rho = o_lines[i][0]; 
		float f_theta = o_lines[i][1];
		Point o_pt1, o_pt2;
		double d_a = cos(f_theta);
		double d_b = sin(f_theta);
		double d_x0 = f_rho * d_a;
		double d_y0 = f_rho * d_b;
		o_pt1.x = cvRound( d_x0 + 3000 * (-d_b) );
		o_pt1.y = cvRound( d_y0 + 3000 * d_a );
		o_pt2.x = cvRound( d_x0 - 3000 * (-d_b) );
		o_pt2.y = cvRound( d_y0 - 3000 * d_a );
		line(o_color_dst, o_pt1, o_pt2, Scalar(0,0,255), 3, CV_AA );
	}
#else
		vector<Vec4i> o_lines;
	HoughLinesP(o_edge_img, o_lines, 1, CV_PI / 180, 30, 0, 100);
	for (size_t i = 0; i < o_lines.size(); i++)
	{
		line(o_color_dst, Point(o_lines[i][0], o_lines[i][1]),
			Point(o_lines[i][2], o_lines[i][3]), Scalar(255, 0, 255), 2, 8);
	}
#endif

	return 0;
}

/*-----------------------------------------------------------------------------
 *
 * function : 检测roi区域的四个顶点
 * 
 * parameter: 
 *		1、Mat o_bin_img  输入的二值图像
 * 	
 *-----------------------------------------------------------------------------
 */
int LaserPointPos::m_roi_point_detect( Mat o_bin_img )
{
	if( o_bin_img.empty() ){
		cout << "the bin image of the funtion m_roi_point_detect() is empty!" << endl;
		return -1;
	}
	if( !om_roi_points.empty() ){
		om_roi_points.clear();
	}

	Mat o_temp_img;
	o_bin_img.copyTo( o_temp_img );
	int n_height = o_bin_img.rows;
	int n_width  = o_bin_img.cols;

	int n_min_dist = 65535;
	int n_temp_dist = 0;
	double d_dist = 0.0;
	Point o_temp_pt(0,0);
	//左上角
	for( int r = 0; r < n_height/2; r++ ){
		for( int c = 0; c < n_width/2; c++ ){

			if( o_temp_img.at<uchar>( r, c ) != 0 ){
				d_dist = ( r - 0 ) * ( r - 0 ) + ( c - 0 ) * ( c - 0 );
				n_temp_dist = int( sqrt( d_dist ) );
				if( n_temp_dist < n_min_dist ){
					n_min_dist = n_temp_dist;

					o_temp_pt.x = c;
					o_temp_pt.y = r;
				}
			}
		}
	}

	om_roi_points.push_back( o_temp_pt );
	n_min_dist = 65535;
	n_temp_dist = 0;
	d_dist = 0.0;
	o_temp_pt = Point(0, 0);
	//左下角
	for( int r = n_height/2; r < n_height; r++ ){
		for( int c = 0; c < n_width/2; c++ ){

			if( o_temp_img.at<uchar>( r, c ) != 0 ){
				d_dist = ( r - n_height ) * ( r - n_height ) + ( c - 0 ) * ( c - 0 );
				n_temp_dist = int( sqrt( d_dist ) );
				if( n_temp_dist < n_min_dist ){
					n_min_dist = n_temp_dist;

					o_temp_pt.x = c;
					o_temp_pt.y = r;
				}
			}
		}
	}

	om_roi_points.push_back( o_temp_pt );
	n_min_dist = 65535;
	n_temp_dist = 0;
	d_dist = 0.0;
	o_temp_pt = Point(0, 0);
	//右下角
	for( int r = n_height/2; r < n_height; r++ ){
		for( int c = n_width/2; c < n_width; c++ ){

			if( o_temp_img.at<uchar>( r, c ) != 0 ){
				d_dist = ( r - n_height ) * ( r - n_height ) + ( c - n_width ) * ( c - n_width );
				n_temp_dist = int( sqrt( d_dist ) );
				if( n_temp_dist < n_min_dist ){
					n_min_dist = n_temp_dist;

					o_temp_pt.x = c;
					o_temp_pt.y = r;
				}
			}
		}
	}

	om_roi_points.push_back( o_temp_pt );
	n_min_dist = 65535;
	n_temp_dist = 0;
	d_dist = 0.0;
	o_temp_pt = Point(0, 0);
	//右上角
	for( int r = 0; r < n_height/2; r++ ){
		for( int c = n_width/2; c < n_width; c++ ){

			if( o_temp_img.at<uchar>( r, c ) != 0 ){
				d_dist = ( r - 0 ) * ( r - 0 ) + ( c - n_width ) * ( c - n_width );
				n_temp_dist = int( sqrt( d_dist ) );
				if( n_temp_dist < n_min_dist ){
					n_min_dist = n_temp_dist;

					o_temp_pt.x = c;
					o_temp_pt.y = r;
				}
			}
		}
	}

	om_roi_points.push_back( o_temp_pt );

 	return 0;
}

/*-----------------------------------------------------------------------------
 *
 * function : roi区域图像校正
 * 
 * parameter: 
 *		1、Mat o_bin_img  输入的图像
 * 	
 *-----------------------------------------------------------------------------
 */
Mat LaserPointPos::m_roi_img_adjust( Mat o_input_img )
{
	if( o_input_img.empty() ){
		cout << "the input image of the funtion m_roi_img_adjust() is empty!" << endl;
		exit(-1);
	}
	m_adjust_laser_pt = Point2f(-1,-1);
	Mat o_input_img_copy;
	o_input_img.copyTo( o_input_img_copy );
	int n_hei = o_input_img_copy.rows;
	int n_wid = o_input_img_copy.cols;
	Mat o_return_img( o_input_img_copy.rows, o_input_img_copy.cols, o_input_img_copy.type() );

	Mat o_warp_mat( 3, 3, CV_32FC1 );
	Point2f o_src_tri[4];
	Point2f o_dst_tri[4];

	o_src_tri[0] = Point2f( float(om_roi_points[0].x), float(om_roi_points[0].y) );
	o_src_tri[1] = Point2f( float(om_roi_points[1].x), float(om_roi_points[1].y) );
	o_src_tri[2] = Point2f( float(om_roi_points[2].x), float(om_roi_points[2].y) );
	o_src_tri[3] = Point2f( float(om_roi_points[3].x), float(om_roi_points[3].y) );

	o_dst_tri[0] = Point2f( 0, 0 );
	o_dst_tri[1] = Point2f( 0, float( n_hei - 0 ) );
	o_dst_tri[2] = Point2f( float( n_wid - 0), float( n_hei - 0 ) );
	o_dst_tri[3] = Point2f( float( n_wid - 0 ), 0 );

	o_warp_mat = getPerspectiveTransform(  o_src_tri, o_dst_tri );

	vector<Point2f> src_points, trans_points;
	src_points.push_back( Point2f(m_laser_point_coords) );
	perspectiveTransform( src_points, trans_points, o_warp_mat );

	m_adjust_laser_pt.x = float( trans_points[0].x / n_wid );
	m_adjust_laser_pt.y = float( trans_points[0].y / n_hei );
	warpPerspective( o_input_img_copy, o_return_img, o_warp_mat, o_return_img.size() );

	if( !src_points.empty() ){
		src_points.clear();
	}

	if( !trans_points.empty() ){
		trans_points.clear();
	}

	return o_return_img;

}





