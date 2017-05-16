// test.cpp : 定义控制台应用程序的入口点。
//

//#include "stdafx.h"
#include "test.h"
#include "stack_point.h"
#include "include\find_corner.h"

//定义输入的图像方向类型
//#define FRONT_IMG
//#define BACK_IMG
//#define LEFT_IMG
#define RIGHT_IMG

//定义全局变量
const int g_ndMaxValue = 100;
const int g_nsigmaColorMaxValue = 200;
const int g_nsigmaSpaceMaxValue = 200;
int g_ndValue;
int g_nsigmaColorValue;
int g_nsigmaSpaceValue;

Mat g_srcImage;
Mat g_dstImage;

//定义回调函数
void on_bilateralFilterTrackbar(int, void*);

int bilateralFilterTest();

void ThresholdProcessing();
void SpatialFunction(double *dstData,int mode_row,int mode_clo,double singa);


int _tmain(int argc, _TCHAR* argv[])
{
	//ThresholdProcessing();

	//bilateralFilterTest();
	//test1();
	//Stack_Point2d_test();
	//Image2double_test();
	cv::Mat A1 = Mat::zeros(2,2,CV_32FC1);
	float *eigen_val = NULL;
	A1.at<float>(0,0) = 3;
	A1.at<float>(0,1) = 5;
	A1.at<float>(1,0) = 5;
	A1.at<float>(1,1) = 7;
	cv::Mat A1_eigen_val ;//= Mat::zeros(2,2,CV_32FC1);
	cv::Mat A1_eigen_vec ;//= Mat::zeros(2,2,CV_32FC1);
	cv::eigen( A1,A1_eigen_val, A1_eigen_vec, 1, 0 );
	//cv::eigen2x2( (float*)A1.data,eigen_val, 
	cout << "A1 = " << endl << A1 << endl;
	cout << "A1_eigen_val = " << endl << A1_eigen_val << endl;
	cout << "A1_eigen_vec = " << endl << A1_eigen_vec << endl;

	double *A2 = new double[2*2];
	double *A2_eigen_vec = new double[2*2];
	double *A2_temp_vec = new double[2*2];
	double *A2_eigen_val = new double[2];

	double A3[2][2] = {3,5,5,7};
	double A3_eigen_vec[2][2] = {0};
	double temp_val = 0;

	memset( (double*)A2_temp_vec, 0, sizeof(double)*4 );
	memset( (double*)A2_eigen_vec, 0, sizeof(double)*4 );
	memset( (double*)A2_eigen_val, 0, sizeof(double)*2 );

	*( A2 + 0 ) = 4.5;
	*( A2 + 1 ) = 7.5;
	*( A2 + 2 ) = 7.5;
	*( A2 + 3 ) = 5.3;

	*(A2_temp_vec + 0) = *( A2 + 0 );
	*(A2_temp_vec + 1) = *( A2 + 1 );
	*(A2_temp_vec + 2) = *( A2 + 2 );
	*(A2_temp_vec + 3) = *( A2 + 3 );

	Eigen_Jacbi( A2, 2, A2_eigen_vec, A2_eigen_val, 0.001, 100 );

	/*if( ( *(A2_eigen_val + 0 ) - *(A2_eigen_val + 1 ) ) > 0.001 )
	{
		temp_val = *( A2_eigen_val + 1 );
		*( A2_eigen_val + 1 ) = *( A2_eigen_val + 0 );
		*( A2_eigen_val + 0 ) = temp_val;
	}

	if( *(A2_eigen_vec+1) > 0 )
	{
		temp_val = *( A2_eigen_vec+0 );
		*( A2_eigen_vec+0 ) = Abs( *( A2_eigen_vec+1 ) );
		*( A2_eigen_vec+1 ) = Abs( temp_val );

		temp_val = *( A2_eigen_vec+3 );
		*( A2_eigen_vec+3 ) = Abs( *( A2_eigen_vec+2 ) ) * (-1);
		*( A2_eigen_vec+2 ) = Abs( temp_val );

	}
	else if( *(A2_eigen_vec+1) < 0 )
	{
		temp_val = *( A2_eigen_vec+0 );
		*( A2_eigen_vec+0 ) = Abs( *( A2_eigen_vec+1 ) );
		*( A2_eigen_vec+1 ) = Abs( temp_val ) * (-1);

		temp_val = *( A2_eigen_vec+3 );
		*( A2_eigen_vec+3 ) = Abs( *( A2_eigen_vec+2 ) ) * (-1);
		*( A2_eigen_vec+2 ) = Abs( temp_val ) * (-1);

	}

	temp_val = ( *( A2_temp_vec+0 ) ) * ( *( A2_eigen_vec+0 ) ) + ( *( A2_temp_vec+1 ) ) * ( *( A2_eigen_vec+2 ) );

	if( Abs( temp_val - *( A2_eigen_vec+0 ) * ( *(A2_eigen_val + 0 ) )  ) > 0.001 )
	{
		*( A2_eigen_vec+0 ) = *( A2_eigen_vec+0 ) * (-1) ;
		*( A2_eigen_vec+3 ) = *( A2_eigen_vec+3 ) * (-1) ;
	}*/





	cout << "A2_eigen_vec = " << endl;
	cout << "[" <<  *(A2_eigen_vec + 0 ) << "," << *(A2_eigen_vec + 1) << "；]" << endl;
	cout << "[" <<  *(A2_eigen_vec + 2 ) << "," << *(A2_eigen_vec + 3) << "；]" << endl;

	cout << "A2_eigen_val = " << endl;
	cout << "[" <<  *(A2_eigen_val + 0 ) << "," << *(A2_eigen_val + 1) << "；]" << endl;

	Jacobi(A3,A3_eigen_vec,100,2);

	cout << "A3_eigen_vec = " << endl;
	cout << "[" <<  A3_eigen_vec[0][0] << "," << A3_eigen_vec[0][1] << ";]" << endl;
	cout << "[" <<  A3_eigen_vec[1][0] << "," << A3_eigen_vec[1][1] << "]" << endl;


	//find_corner_test();
	delete []A2_eigen_val;
	delete []A2_eigen_vec;
	delete []A2_temp_vec;
	return 0;

	int hist[14] = {1,24,2,4,5,7,23,7,23,12,6,5,34,27};
	int hist_2[14] = {1,24,2,4,5,7,23,7,23,12,6,5,34,27};

	std::vector<float> src_hist;
	std::vector<float> src_hist_1;

	std::vector<float> dst_hist;
	std::vector<float> mode_col_idx; //模式的索引
	std::vector<float> mode_col_val; //模式的值

	for( int i = 0; i < 14; i++ )
	{
		src_hist.push_back( (float)hist[i] );
		src_hist_1.push_back( (float)i );
	}

	hist_smooth( src_hist, dst_hist, 1.0 );
	mode_find( dst_hist, mode_col_idx, mode_col_val );

	sort_test( hist_2, 14 );

	sort_mode( src_hist_1,src_hist );

	std::cout << "dst_hist:" << std::endl ;
	for( int i = 0; i < dst_hist.size(); i++)
	{
		std::cout << dst_hist.at(i) << "\t";
	}

	std::cout << "\src_hist:" << std::endl ;
	for( int i = 0; i < src_hist.size(); i++)
	{
		std::cout << src_hist.at(i) << "\t";
	}

	std::cout << "\src_hist_1:" << std::endl ;
	for( int i = 0; i < src_hist_1.size(); i++)
	{
		std::cout << src_hist_1.at(i) << "\t";
	}

    return 0;

	cv::String src_img_name;
	cv::String mask_zone_img_name;
	cv::String save_mask_img_name;
	cv::String dir_path = "D:/project/test/test/ChangAn_2/";

	//cv::String src_img_path = "D:/project/cal_ground_local/cal_ground_local/test_img/changchunyingjiadata/H_left1000.bmp";

	//const std::string s_dir_path = "./chess_img_2/";

#ifdef FRONT_IMG
	src_img_name = "Front.bmp";
	mask_zone_img_name = "Front_mask_region.bmp";
	save_mask_img_name = "Front_mask.bmp";
#endif

#ifdef BACK_IMG
	src_img_name = "Back.bmp";
	mask_zone_img_name = "Back_mask_region.bmp";
	save_mask_img_name = "Back_mask.bmp";
#endif

#ifdef LEFT_IMG
	src_img_name = "Left.bmp";
	mask_zone_img_name = "Left_mask_region.bmp";
	save_mask_img_name = "Left_mask.bmp";
#endif

#ifdef RIGHT_IMG
	src_img_name = "Right.bmp";
	mask_zone_img_name = "Right_mask_region.bmp";
	save_mask_img_name = "Right_mask.bmp";
#endif

	src_img_name = dir_path + src_img_name;
	mask_zone_img_name = dir_path + mask_zone_img_name;
	save_mask_img_name = dir_path + save_mask_img_name;

	//int i = 1;
	//std::stringstream ss;
	//std::string img_file_name;
	//cv::Size board_size = cv::Size(6,9);   //标定板每行，每列角点数

	//while(i < 18)
	//{
	//	ss << s_dir_path <<"img" << i << ".jpg";
	//	ss >> img_file_name;
	//	std::cout << "process " << img_file_name << std::endl;
	//	if( i == 15 )
	//		board_size = cv::Size(6,8);

	//	if( i == 16 )
	//		board_size = cv::Size(10,15);

	//	chessboard_corner_detect_test( img_file_name, board_size );
	//	ss.clear();
	//	ss.str("");
	//	i++;
	//}
	get_mask_img(src_img_name,mask_zone_img_name,save_mask_img_name);
	
 	return 0;
}

void get_mask_img(const cv::String src_img_path, const cv::String mask_zone_img_path, const cv::String save_mask_img_path )
{
	// 读入图片
	if(src_img_path.empty() || mask_zone_img_path.empty() || save_mask_img_path.empty() ) 
	{
		std::cout << "input image path is fail!" << std::endl;
		return;
	}
	Mat src_img = imread(src_img_path,IMREAD_GRAYSCALE); //加载原始图像
	Mat mask_img = imread(mask_zone_img_path,IMREAD_GRAYSCALE);//加载mask区域的图像，除了mask区域为白色的其他区域为原图像的
	//cv::vector<cv::Point2f> pointfs;
	//均值滤波
	blur(src_img,src_img,Size(5,5),Point(-1,-1));
	blur(mask_img,mask_img,Size(5,5),Point(-1,-1));

	Mat save_img(src_img.rows,src_img.cols,CV_8UC1) ;
	Mat sub_img = abs(src_img - mask_img);

	namedWindow("src_img");
	namedWindow("mask_img");
	namedWindow("sub_img");
	namedWindow("save_img");

	imshow("src_img",src_img);
	imshow("mask_img",mask_img);
	imshow("sub_img",sub_img);
	
	int nheight = src_img.rows;
	int nwidth  = src_img.cols;

	for(int r = 0; r < nheight; r++)
	{
		for(int c = 0; c < nwidth; c++)
		{
			 if(sub_img.at<uchar>(r,c) < 10){
				save_img.at<uchar>(r,c) = 0;
			 }
			 else
				 save_img.at<uchar>(r,c) = 255;
		}
	}
	imshow("save_img",save_img );
	imwrite( save_mask_img_path,save_img );

	//Mat img5 = imread("D:/project/test/test/H_right1000.bmp",IMREAD_GRAYSCALE);
	//Mat mv[3];
	//split(img2,mv);
	// 创建一个名为 
	
	/*cvNamedWindow("R");
	cvNamedWindow("G");
	cvNamedWindow("B");
	imshow("R",mv[0]);
	imshow("G",mv[1]);
	imshow("B",mv[2]);
	*/

	waitKey(0);
	destroyAllWindows();
}

/***************************************************************
function：		棋盘角点检测测试
----------------------------------------------------------------
参数类型			参数					I/O类型		参数说明
const cv::String	src_img_path			in			输入要检测角点的图像路径
---------------------------------------------------------------
fan-in：		top_test()或者main()
fan-out：		
****************************************************************/
void chessboard_corner_detect_test( const cv::String src_img_path, const cv::Size board_size )
{
	cv::Mat image;
	cv::Mat Extractcorner;
    std::vector<cv::Point2f> corners;    //用来储存所有角点坐标
    //cv::Size board_size = cv::Size(6,9);   //标定板每行，每列角点数
    image = cv::imread(src_img_path);
    Extractcorner = image.clone();

    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, CV_RGB2GRAY);
    bool patternfound = cv::findChessboardCorners(image, board_size, corners, cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
    //if (!patternfound)
    //{
    //    std::cout << "can not find chessboard corners!" << std::endl;
    //    exit(1);
    //}
    //else
    //{
    //    //亚像素精确化
    //    cv::cornerSubPix(imageGray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    //}

    //角点检测图像显示
    for (int i = 0; i < corners.size(); i++)
    {
        cv::circle(Extractcorner, corners[i], 3, cv::Scalar(255, 0, 255), 2);
    }
    cv::imshow("Extractcorner", Extractcorner);

    cv::waitKey(0);
	cv::destroyAllWindows();
}

void on_bilateralFilterTrackbar(int, void*)
{
    bilateralFilter(g_srcImage, g_dstImage, g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue);
    imshow("BilateralFilterImage", g_dstImage);
}

int bilateralFilterTest()
{
	g_srcImage = imread("./test_img2/H_left1000.bmp");

    //判断图像是否加载成功
    if(g_srcImage.empty())
    {
        std::cout << "Image load fail!" << std::endl;
        return -1;
    }
    else
        std::cout << "Image load success!" << std::endl << std::endl;

    namedWindow("srcImg", WINDOW_AUTOSIZE);
    imshow("srcImg", g_srcImage);

    //定义输出图像窗口属性和轨迹条属性
    namedWindow("BilateralFilterImage", WINDOW_AUTOSIZE);
    g_ndValue = 10;
    g_nsigmaColorValue = 10;
    g_nsigmaSpaceValue = 10;

	std::stringstream ss;

    std::string dName;
    //sprintf(dName, "neighbourhood diameter =  %d", g_ndMaxValue); //邻域直径

	ss << "neighbourhood diameter " << g_ndMaxValue;
	ss >> dName;
	ss.clear();
	ss.str("");

    std::string sigmaColorName;
    //sprintf(sigmaColorName, "sigmaColor %d", g_nsigmaColorMaxValue);

	ss << "sigmaColor " << g_nsigmaColorMaxValue;
	ss >> sigmaColorName;
	ss.clear();
	ss.str("");

    std::string sigmaSpaceName;
   // sprintf(sigmaSpaceName, "sigmaSpace %d", g_nsigmaSpaceMaxValue);
	ss << "sigmaSpace " << g_nsigmaSpaceMaxValue;
	ss >> sigmaSpaceName;
	ss.clear();
	ss.str("");
    //创建轨迹条
    createTrackbar(dName, "BilateralFilterImage", &g_ndValue, g_ndMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_ndValue, 0);

    createTrackbar(sigmaColorName, "BilateralFilterImage", &g_nsigmaColorValue,
                     g_nsigmaColorMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_nsigmaColorValue, 0);

    createTrackbar(sigmaSpaceName, "BilateralFilterImage", &g_nsigmaSpaceValue,
                    g_nsigmaSpaceMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_nsigmaSpaceValue, 0);

    waitKey(0);
	imwrite( "BilateralFilterImage.jpg",g_dstImage );

    return 0;
}

/***************************************************************
function：		对图像进行阈值化处理
----------------------------------------------------------------
无参型函数
---------------------------------------------------------------
fan-in：		top_test()或者main()
fan-out：		
****************************************************************/
void ThresholdProcessing()
{
	cv::Mat srcImg;
	cv::Mat maskImg;

	int nThreshValueHigh = 130;
	int nThreshValueLow = 70;

	srcImg = imread( "D:/project/test/test/changchunyingjia/H_left1000.bmp", IMREAD_GRAYSCALE );
	maskImg = imread( "D:/project/test/test/changchunyingjia/left_mask_image.bmp", IMREAD_GRAYSCALE );
	

	cv::Mat filterImg;
	bilateralFilter(srcImg,filterImg,5,10.0,2.0);//这里滤波没什么效果，不明白

	cv::Mat RoiImg = filterImg.clone();

	int nheight = srcImg.rows;
	int nwidth  = srcImg.cols;

	for(int r = 0; r < nheight; r++)
	{
		for(int c = 0; c < nwidth; c++)
		{
			 if(maskImg.at<uchar>(r,c) < 80)
			 {
				RoiImg.at<uchar>(r,c) = 0;
			 }
		}
	}


	for(int r = 0; r < nheight; r++)
	{
		for(int c = 0; c < nwidth; c++)
		{

			//if( r > 245 && c > 550) 
			//{
				 if(RoiImg.at<uchar>(r,c) <= 75 && RoiImg.at<uchar>(r,c) >= 2 )
				 {
					RoiImg.at<uchar>(r,c) = 60;
				 }

				 if( RoiImg.at<uchar>(r,c) >= 76 )
				 {
					RoiImg.at<uchar>(r,c) = 130;
				 }
			//}
		}
	}

	imwrite("RoiImg.bmp",RoiImg);



	namedWindow("srcImg", WINDOW_AUTOSIZE);
	namedWindow("filterImg",WINDOW_AUTOSIZE);
	namedWindow("RoiImg",WINDOW_AUTOSIZE);

	imshow("srcImg",srcImg);
	imshow("filterImg",filterImg);
	imshow("RoiImg",RoiImg);

	waitKey(0);

	destroyAllWindows();

}

void SpatialFunction(double *dstData,int mode_row,int mode_clo,double singa)
{
	int i,j;

	//double **ret = (double**)malloc(mode_row*mode_clo*sizeof(double));
	if(!dstData)
	{
		return;
	}

	for (i = -(mode_row/2); i <= (mode_row/2); i++)
	{
		for (j = -(mode_clo/2); j <= (mode_clo/2); j++)
		{
			dstData[(i+mode_row)*mode_clo+(j+mode_clo)] = exp(-1/2 * (i * i + j * j) / (singa*singa));
		}
	}

	//return ret;

}

void test1()
{
	
	int mode_size = 5;
	double spa_singa = 2.0,range_singa = 2.0;
	char *infile;
	//char *outfile = "lena_1.bmp"; 
	
// 	double **in_image = (double **)CComlib::fspace_2d(row,clo,sizeof(double));
// 	double **out_imge = (double **)CComlib::fspace_2d(row,clo,sizeof(double));

	Mat srcImg = imread( "D:/project/test/test/ChangAn_2/Front.bmp", IMREAD_COLOR);

	//BYTE *imge = (BYTE *)FileIO::Read8BitBmpFile2Img("2.bmp",&clo,&row);
	int row = srcImg.rows;
	int col = srcImg.cols;
	//double **in_image = (double **)fspace_2d(row,col,sizeof(double));
	//double **out_imge = (double **)fspace_2d(row,col,sizeof(double));

	cv::Mat filterImg;
	bilateralFilter(srcImg,filterImg,5,10.0,2.0);//这里滤波没什么效果，不明白

	//cv::resize(filterImg, dst, cv::Size(resize_width, resize_height), (0, 0), (0, 0), cv::INTER_LINEAR);

	unsigned char *srcImg_data = srcImg.data;

	Mat ResizeImg( (int)row*2, (int)col*2, CV_8UC1 );
	Mat ResizeImg_2( (int)row*1.5, (int)col*1.5, CV_8UC3 );
	Mat ResizeImg_3( (int)row*1.5, (int)col*1.5, CV_8UC3 );
	
	unsigned char *p_dstimg_data = ResizeImg_3.data;

	unsigned char *ResizeImg_data = ResizeImg.data;
	int n_row = ResizeImg.rows;
	int n_col = ResizeImg.cols;

	cv::resize(filterImg, ResizeImg_2, cv::Size(n_col, n_row), (0, 0), (0, 0), cv::INTER_LINEAR);//CV_INTER_AREA :在缩小是效果比较好，INTER_LINEAR：在放大时效果比较好
   // ResizeImage(srcImg_data,col,row,p_dstimg_data,n_col, n_row);
	//Resize_image_3();
	Resize_image_4();
	//Resize_image_2( srcImg_data, p_dstimg_data, col, row, ResizeImg_3.cols, ResizeImg_3.rows );
	Resize_image_4( srcImg_data, (int)srcImg.step[0], p_dstimg_data, (int)ResizeImg_3.step[0], srcImg.cols, srcImg.rows, ResizeImg_3.cols, ResizeImg_3.rows, (int)srcImg.step[1] );
	if(!srcImg.data)
	{
		printf("Error,failed read!");
		// 		break;
	}
	
	//for(int i = 0; i < srcImg.rows; i++)
	//	for(int j = 0; j < srcImg.cols; j++)
	//	{
	//		in_image[i][j] = srcImg.at<uchar>(i,j);
	//		
	//	}

	//for( int r = 0; r < n_row; r++)
	//{
	//	for( int c = 0; c < n_col; c++)
	//	{

	//	}
	//}

	//for( int r = 0; r < n_row; r++)
 //   {
 //        const uchar *inData = filterImg.ptr<uchar>(r/2);
 //        uchar *outData = ResizeImg.ptr<uchar>(r);

 //        for( int c = 0; c < n_col; c++)
 //        {
 //            *(outData++) = inData[c/2];
 //        }
 //   }

	
	//free((void*) imge);
	
	//Bilater_Filter_Test(out_imge,in_image,row,col,mode_size,spa_singa,range_singa);

	//Mat dstImg(row,col,CV_8UC1);

	//for(int m = 0; m < row; m++)
	//	for(int n = 0; n < col; n++)
	//		dstImg.at<uchar>(m,n) = out_imge[m][n];

	//ffree_2d( (void **)out_imge,row );
	//ffree_2d( (void **)in_image,row );

	namedWindow("filterImg",WINDOW_AUTOSIZE);
	namedWindow("srcImg",WINDOW_AUTOSIZE);
	//namedWindow("dstImg",WINDOW_AUTOSIZE);
	namedWindow("ResizeImg",WINDOW_AUTOSIZE);
	namedWindow("ResizeImg_2",WINDOW_AUTOSIZE);
	imshow("ResizeImg_2",ResizeImg_2);
	imwrite("D:/project/test/test/ChangAn_2/2.bmp",ResizeImg_2);
	namedWindow("ResizeImg_3",WINDOW_AUTOSIZE);
	imshow("ResizeImg_3",ResizeImg_3);
	imshow("ResizeImg",ResizeImg);
	imwrite("D:/project/test/test/ChangAn_2/3.bmp",ResizeImg);
	imshow("srcImg",srcImg);
	//imshow("dstImg",dstImg);
	imshow("filterImg",filterImg);

	//imwrite("D:/project/test/test/ChangAn_2/Filted.bmp",dstImg);

	waitKey(0);
	return;
	//delete [] output_data;

}

void Bilater_Filter_Test(double **output_img,double **in_data,int row,int clo,int size,double spa_singa,double range_singa)
{
	int i,j,k,l;
// 	double sum1 = 0.0;
// 	double sum2 = 0.0;
	double temp1,temp2,temp;
	int mode_row =  size;
	int mode_clo = size;

// 	double **temp1 = (double **)CComlib::fspace_2d(mode_row,mode_clo,sizeof(double));
	//double **output_img = (double **)CComlib::fspace_2d(row,clo,sizeof(double));

	for (i = 0; i < row; i++)
	{	
		for(j = 0; j <clo; j++)
		{
			double sum1 = 0.0;
			double sum2 = 0.0;
// 			temp1 = SpatialFunction(mode_row,mode_clo,spa_singa);

			for(k = -(mode_row / 2); k <= (mode_row / 2); k++)
			{	
				for (l = -(mode_clo / 2); l <= (mode_clo / 2); l++)
				{
					//判断是否越出边界
					int p = i+k;
					if (p < 0)	p = -p -1;
					else if(p >= row)	p = 2 * row - p -1;

					int q = j+l;
					if(q < 0)	q = -q -1;
					else if(q >= clo)	q = 2 *clo -q -1;


					double med1 = double(abs((k * k + l * l) / (spa_singa*spa_singa)));
					temp1 = double(exp((-0.5) * med1));
// 					double dkd = double(-0.5);
// 					printf("%f  ",med1);
// 					temp2 = RangeWeightFunction(in_data[p][q],in_data[i][j],range_singa);
					double med2 = double(abs((in_data[p][q] - in_data[i][j])*(in_data[p][q] - in_data[i][j]) / (range_singa*range_singa)));
					temp2 = double(exp((-0.5) * med2));
					
// 					double dem = abs(in_data[p][q] - in_data[i][j]);
// 					printf("%f  ",abs(in_data[p][q] - in_data[i][j]));
// 					printf("%f  ",med2);
// 




					temp = temp1 * temp2;
// 					printf("%f  ",temp);

					sum1 = sum1 + temp;
// 					printf("%f  ",sum1);

					sum2 = sum2 + temp * in_data[p][q];
// 					printf("%f\n",sum2);


				}
// 				printf("\n");// 
			}

			output_img[i][j] = sum2 / sum1;
// 			printf("%f\n",in_data[i][j]);


		}
	}
	//return output_img;

// 		CComlib::ffree_2d((void**)temp1,mode_row);
// 		CComlib::ffree_2d((void**)input_img,row);
}



/*****************************   fspace_1d  **************************/
/*  To allocation a 1_dimension dynamic array  */
void * fspace_1d(int col, int length)
{
    void *b;
    
    b = (void *)calloc(length,col);
    
    if(!b) return NULL;
    
    return(b);
}

/*************      fspace_2d    ***************************************/
/*  To allocation a 2_dimension dynamic array  */
void ** fspace_2d(int row,int col,int lenth)
{
    int i;
    void **b;
    
    b = (void **)calloc(sizeof(void *),row);
	
    if(!b) return NULL;
    
    for(i=0;i<row;i++)
    {
        b[i] = (void *)calloc(lenth,col);
        if(!b[i]) return NULL;
    }
    
    return(b);
}

/*************      fspace_3d    ***************************************/
/*  To allocation a 3_dimension dynamic array  */
void *** fspace_3d(int row1,int row2,int row3,int lenth)
{
    int i;
    void ***b;
    
    b = (void ***)calloc(sizeof(void **),row1);
    
    if(!b) return NULL;
    
    for(i=0;i<row1;i++)
        b[i] = (void **)fspace_2d(row2,row3,lenth);
    
    return(b);
}

/*******************************  ffree_1d   ****************************/
/*  To free a 1_dimension dynamic array  */
void ffree_1d(void *a)
{
    if(a==NULL) return;
    
    free(a);
    
    a = NULL;
}

/*******************************  ffree_2d   ****************************/
/*  To free a 2_dimension dynamic array  */
void ffree_2d(void **a,int row)
{
    if(a==NULL) return ;
    
    int i;
    
    for(i=0;i<row;i++) { free(a[i]); a[i]=NULL; }
    free(a);
    
    a = NULL;
}

/*******************************  ffree_3d   ****************************/
/*  To free a 3_dimension dynamic array  */
void ffree_3d(void ***a,int row1,int row2)
{
    if(a==NULL) return ;
    int i;
    
    for(i=0;i<row1;i++)
        ffree_2d((void **)a[i],row2);
    free(a);
    
    a = NULL;
}

///////////////////////////////////缩放图像
static void _ieInterpImageBilinear8UC1_Ver3_RowFilter(unsigned char* src, long* dst, int len, int* leftIdx, int* rightIdx, long* weight, int shift)
{
    int i;
    for(i = 0; i < len - 4; i+=4) {
        *dst++ = ((1<<shift) - weight[i])*src[leftIdx[i]] + weight[i]*src[rightIdx[i]];
        *dst++ = ((1<<shift) - weight[i+1])*src[leftIdx[i+1]] + weight[i+1]*src[rightIdx[i+1]];
        *dst++ = ((1<<shift) - weight[i+2])*src[leftIdx[i+2]] + weight[i+2]*src[rightIdx[i+2]];
        *dst++ = ((1<<shift) - weight[i+3])*src[leftIdx[i+3]] + weight[i+3]*src[rightIdx[i+3]];
  
    }
    for( ; i < len; ++i) {
        *dst++ = ((1<<shift) - weight[i])*src[leftIdx[i]] + weight[i]*src[rightIdx[i]];
    }
}

#define IET_MAX(x,y) (x)>(y)?(x):(y)
#define IET_MIN(x,y) (x)>(y)?(y):(x)
#define IET_SWAP(x,y,tmp) (tmp)=(x);(x)=(y);(y)=(tmp);

static void ResizeImage(unsigned char* pSrc,int src_w,int src_h,
                        unsigned char* pDst,int dst_w, int dst_h)
{
    int i, j;
    int sw, sh, sstep;
    int dw, dh, dstep;
    unsigned char *sdata, *ddata;
    float horScaleRatio, verScaleRatio;
    long *rowBuf1, *rowBuf2;
    long *upLinePtr, *downLinePtr, *tempPtr;
    long *horWeight;
    int *horLeftIdx, *horRightIdx;
    int preVerUpIdx, preVerDownIdx;
    int shift = 8;
  
    sw = src_w;
    sh = src_h;
    sstep=24;
    sdata=pSrc;
    dw=dst_w;
    dh=dst_h;
    dstep=24;
    ddata=pDst;
  
    horScaleRatio = sw / (float)(dw);
    verScaleRatio = sh / (float)(dh);
  
    rowBuf1 = new long[dw];
    rowBuf2 = new long[dw];
    horWeight = new long[dw];
    horLeftIdx = new int[dw];
    horRightIdx = new int[dw];
  
  
    //col interpolation
  
    //计算目标图像像素横向的左右邻居序号，和权重。
    for(i = 0; i < dw; i++) {
        float pos = (i + 0.5f) * horScaleRatio;
        horLeftIdx[i] = (int)(IET_MAX(pos - 0.5f, 0));
        horRightIdx[i] = (int)(IET_MIN(pos + 0.5f, sw-1));
        horWeight[i] = (long) (fabs(pos - 0.5f - horLeftIdx[i]) * (1<<shift));
    }
  
    preVerUpIdx = -1;
    preVerDownIdx = -1;
    upLinePtr = rowBuf1;
    downLinePtr = rowBuf2;
    for(j = 0; j < dh; j++) {
        float pos = (j + 0.5f) * verScaleRatio;
        int verUpIdx = (int)(IET_MAX(pos - 0.5f, 0));
        int verDownIdx = (int)(IET_MIN(pos + 0.5f, sh-1));
        long verWeight = (long) (fabs(pos - 0.5f - verUpIdx) * (1<<shift));
  
        if(verUpIdx == preVerUpIdx && verDownIdx == preVerDownIdx) {
            ;
            //do nothing
        } else if(verUpIdx == preVerDownIdx) {
            IET_SWAP(upLinePtr, downLinePtr, tempPtr);
            _ieInterpImageBilinear8UC1_Ver3_RowFilter(sdata + sstep*verDownIdx, downLinePtr, dw, horLeftIdx, horRightIdx, horWeight, shift);
        } else {
            _ieInterpImageBilinear8UC1_Ver3_RowFilter(sdata + sstep*verUpIdx,   upLinePtr, dw, horLeftIdx, horRightIdx, horWeight, shift);
            _ieInterpImageBilinear8UC1_Ver3_RowFilter(sdata + sstep*verDownIdx, downLinePtr, dw, horLeftIdx, horRightIdx, horWeight, shift);
        }
  
        unsigned char* _ptr = ddata + dstep*j;
        for(i = 0; i < dw-4; i+=4) {
            *_ptr++ = (unsigned char) ( (((1 << shift) - verWeight)*upLinePtr[i] + verWeight*downLinePtr[i]) >> (2*shift) );
            *_ptr++ = (unsigned char) ( (((1 << shift) - verWeight)*upLinePtr[i+1] + verWeight*downLinePtr[i+1]) >> (2*shift) );
            *_ptr++ = (unsigned char) ( (((1 << shift) - verWeight)*upLinePtr[i+2] + verWeight*downLinePtr[i+2]) >> (2*shift) );
            *_ptr++ = (unsigned char) ( (((1 << shift) - verWeight)*upLinePtr[i+3] + verWeight*downLinePtr[i+3]) >> (2*shift) );
        }
        for(; i < dw; i++) {
            *_ptr++ = (unsigned char) ( (((1<<shift) - verWeight)*upLinePtr[i] + verWeight*downLinePtr[i]) >> (2*shift) );
        }
        preVerUpIdx = verUpIdx;
        preVerDownIdx = verDownIdx;
    }
    delete []rowBuf1;
    delete []rowBuf2;
    delete []horWeight;
    delete []horLeftIdx;
    delete []horRightIdx;
}

void Resize_image_2( const unsigned char *p_srcimg_data, unsigned char *p_dstimg_data, const int n_srcimg_w, const int n_srcimg_h, const int n_dstimg_w, const int n_dstimg_h )
{
	int n_w0 = n_srcimg_w;
	int n_h0 = n_srcimg_h;

	int n_w1 = n_dstimg_w;
	int n_h1 = n_dstimg_h;

	//if( n_w0 <= 1 || n_w1 <= 1 || n_h0 <= 1 || n_h1 <= 1 )
	//{
	//	printf("the size of the image(src or dst) is error!\n");
	//	//assert(n_w0 > 1);
	//	return;
	//}

	float f_scale_factor_w = (float)(n_w0)/(float)n_w1;
	float f_scale_factor_h = (float)(n_h0)/(float)n_h1;

	float f_x0 = 0, f_y0 = 0;
	int n_x1 = 0;
	int n_x2 = 0;
	int n_y1 = 0;
	int n_y2 = 0;

	float f_y1 = 0;
	float f_y2 = 0;

	int *pn_array_x1 = (int *)calloc(n_w1,sizeof(int));
	int *pn_array_x2 = (int *)calloc(n_w1,sizeof(int));
	float *pn_array_sub_x1 = (float *)calloc(n_w1,sizeof(float));
	float *pn_array_sun_x2 = (float *)calloc(n_w1,sizeof(float));

	float f_x1 = 0;
	float f_x2 = 0;

	float f_s1 = 0;
	float f_s2 = 0;
	float f_s3 = 0;
	float f_s4 = 0;

	float f_grayvalue;

	int n_col,n_row;

	for(n_col = 0; n_col < n_w1; n_col++)
	{
		f_x0 = n_col * f_scale_factor_w;
		pn_array_x1[n_col] = (int)(f_x0);
		pn_array_x2[n_col] = std::min(((int)(f_x0+1)),(n_w1-1));
		pn_array_sub_x1[n_col] = f_x0 - pn_array_x1[n_col];
		pn_array_sun_x2[n_col] = 1.0 - pn_array_sub_x1[n_col];
	}

	for(n_row = 0; n_row < n_h1; n_row++)
	{
		f_y0 = (float)n_row * f_scale_factor_h;
		n_y1 = (int)(f_y0);
		n_y2 = std::min(((int)(f_y0+1)),(n_h1-1));
		f_y1 = f_y0 - n_y1; //p
		f_y2 = 1.0 - f_y1; //1-p

		for(n_col = 0; n_col < n_w1; n_col++)
		{
			n_x1 = pn_array_x1[n_col];
			n_x2 = pn_array_x2[n_col];
			f_x1 = pn_array_sub_x1[n_col]; //q
			f_x2 = pn_array_sun_x2[n_col]; //1-q

			f_s1 = f_x2 * f_y2; // 1-q * 1-p
			f_s2 = f_x1 * f_y2; // q * 1-p
			f_s3 = f_x1 * f_y1; // q * p
			f_s4 = f_x2 * f_y1; // 1-q * p

			f_grayvalue = p_srcimg_data[n_y1 * n_w0 + n_x1] * f_s1;
			f_grayvalue += p_srcimg_data[n_y1 * n_w0 + n_x2] * f_s2;
			f_grayvalue += p_srcimg_data[n_y2 * n_w0 + n_x1] * f_s4;
			f_grayvalue += p_srcimg_data[n_y2 * n_w0 + n_x2] * f_s3;

			p_dstimg_data[n_row * n_w1 + n_col] = (int)f_grayvalue;

			/*p_dstimg_data[n_row * n_w1 + n_col] = (int)(p_srcimg_data[n_y1 * n_w0 + n_x1] * f_s1 + p_srcimg_data[n_y2 * n_w0 + n_x1] * f_s2 + \
														  p_srcimg_data[n_y1 * n_w0 + n_x2] * f_s4 + p_srcimg_data[n_y2 * n_w0 + n_x2] * f_s3 );*/

		}
	}


	free(pn_array_x1);
	free(pn_array_x2);
	free(pn_array_sub_x1);
	free(pn_array_sun_x2);


}

void Resize_image_3()
{
	//printf("%s\n",argv[1]);  
      
    IplImage *pSrcImg = cvLoadImage("D:/project/test/test/ChangAn_2/Front.bmp",CV_LOAD_IMAGE_COLOR);  
    if(!pSrcImg)  
    {  
        printf("Load Image failed!\n");  
        return ;  
    }  
  
	int dstImgHeight = (int)(pSrcImg->height * 0.2);   
	int dstImgWidth = (int)(pSrcImg->width * 0.2);  
      
    IplImage *pDstImg = cvCreateImage(cvSize(dstImgWidth,dstImgHeight),pSrcImg->depth,3);  
      
    unsigned char * srcImgData = (unsigned char *)pSrcImg->imageData;  
    unsigned char * dstImgData = (unsigned char *)pDstImg->imageData;  
      
    int srcImgHeight = pSrcImg->height;  
    int srcImgWidth = pSrcImg->width;  
    int srcImgWidthStep = pSrcImg->widthStep;  
    int dstImgWidthStep = pDstImg->widthStep;  
  
    float heightRatio = (float)dstImgHeight/(float)srcImgHeight;  
    float widthRatio = (float)dstImgWidth/(float)srcImgWidth;  
  
    for (int i=0;i<dstImgHeight;i++)  
    {  
        float fx = (float)i/heightRatio;  
        int nx = (int)fx;  
        int nxa1 = nx+1;  
        float p = fx-nx;   
        if (nxa1>=srcImgHeight) //最后一行  
        {  
            nxa1 = srcImgHeight-1;  
        }  
        for (int j=0;j<dstImgWidth;j++)  
        {  
            float fy = (float)j/widthRatio;  
            int ny = (int)fy;  
            int nya1 = ny+1;  
            float q = fy - ny;  
            if (nya1>=srcImgWidth) //该行最后一个元素  
            {  
                nya1 = srcImgWidth - 1;  
            }  
  
            float b = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3];  
            b += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3];  
            b += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3];  
            b += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3];  
  
            dstImgData[i*dstImgWidthStep+j*3]=(int)b;  
  
            float g = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3+1];  
            g += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3+1];  
            g += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3+1];  
            g += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3+1];  
  
            dstImgData[i*dstImgWidthStep+j*3+1]=(int)g;  
  
            float r = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3+2];  
            r += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3+2];  
            r += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3+2];  
            r += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3+2];  
  
            dstImgData[i*dstImgWidthStep+j*3+2]=(int)r;  
        }  
    }  
      
    cvSaveImage("D:/project/test/test/ChangAn_2/1.bmp",pDstImg);  
    cvNamedWindow("SrcImage",CV_WINDOW_AUTOSIZE);  
    cvNamedWindow("DstImage",CV_WINDOW_AUTOSIZE);  
    // 显示图像pSrc  
    cvShowImage("SrcImage",pSrcImg);  
    cvShowImage("DstImage",pDstImg);  
    cvWaitKey(0);  
    cvDestroyAllWindows();  
    cvReleaseImage(&pSrcImg);  
    cvReleaseImage(&pDstImg);  
}

void Resize_image_4()
{
	IplImage *pSrcImg = cvLoadImage("D:/project/test/test/ChangAn_2/Front.bmp",CV_LOAD_IMAGE_GRAYSCALE);  //CV_LOAD_IMAGE_GRAYSCALE,CV_LOAD_IMAGE_COLOR
    if(!pSrcImg)  
    {  
        printf("Load Image failed!\n");  
        return ;  
    }  
	
	float f_scale_factor = 0.2;

	int dstImgHeight = (int)(pSrcImg->height * f_scale_factor );   
	int dstImgWidth = (int)(pSrcImg->width * f_scale_factor );  
      
	IplImage *pDstImg = cvCreateImage(cvSize(dstImgWidth,dstImgHeight),pSrcImg->depth,pSrcImg->nChannels);  
      
    unsigned char * srcImgData = (unsigned char *)pSrcImg->imageData;  
    unsigned char * dstImgData = (unsigned char *)pDstImg->imageData;  
      
    int srcImgHeight = pSrcImg->height;  
    int srcImgWidth = pSrcImg->width;  
    int srcImgWidthStep = pSrcImg->widthStep;  
    int dstImgWidthStep = pDstImg->widthStep;  
  
    float heightRatio = (float)dstImgHeight/(float)srcImgHeight;  
    float widthRatio = (float)dstImgWidth/(float)srcImgWidth;  


  
    for (int i=0;i<dstImgHeight;i++)  
    {  
        float fx = (float)i/heightRatio;  
        int nx = (int)fx;  
        int nxa1 = nx+1;  
        float p = fx-nx;   
        if (nxa1>=srcImgHeight) //最后一行  
        {  
            nxa1 = srcImgHeight-1;  
        }  
        for (int j=0;j<dstImgWidth;j++)  
        {  
            float fy = (float)j/widthRatio;  
            int ny = (int)fy;  
            int nya1 = ny+1;  
            float q = fy - ny;  
            if (nya1>=srcImgWidth) //该行最后一个元素  
            {  
                nya1 = srcImgWidth - 1;  
            }  
  
			if(1 == pSrcImg->nChannels){

				float b = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny];  
					  b += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1];  
					  b += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny];  
					  b += p*q*srcImgData[nxa1*srcImgWidthStep+nya1];  
  
				dstImgData[i*dstImgWidthStep+j]=(int)b;

			}else if(3 == pSrcImg->nChannels){

				float b = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3];  
				b += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3];  
				b += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3];  
				b += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3];  
  
				dstImgData[i*dstImgWidthStep+j*3]=(int)b;  
  
				float g = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3+1];  
				g += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3+1];  
				g += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3+1];  
				g += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3+1];  
  
				dstImgData[i*dstImgWidthStep+j*3+1]=(int)g;  
  
				float r = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3+2];  
				r += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3+2];  
				r += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3+2];  
				r += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3+2];  
  
				dstImgData[i*dstImgWidthStep+j*3+2]=(int)r; 
			}
           /* float b = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3];  
            b += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3];  
            b += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3];  
            b += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3];  
  
            dstImgData[i*dstImgWidthStep+j*3]=(int)b;  
  
            float g = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3+1];  
            g += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3+1];  
            g += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3+1];  
            g += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3+1];  
  
            dstImgData[i*dstImgWidthStep+j*3+1]=(int)g;  
  
            float r = (1-p)*(1-q)*srcImgData[nx*srcImgWidthStep+ny*3+2];  
            r += (1-p)*q*srcImgData[nx*srcImgWidthStep+nya1*3+2];  
            r += p*(1-q)*srcImgData[nxa1*srcImgWidthStep+ny*3+2];  
            r += p*q*srcImgData[nxa1*srcImgWidthStep+nya1*3+2];  
  
            dstImgData[i*dstImgWidthStep+j*3+2]=(int)r;  */
        }  
    }  
      
    cvSaveImage("D:/project/test/test/ChangAn_2/1.bmp",pDstImg);  
    cvNamedWindow("SrcImage",CV_WINDOW_AUTOSIZE);  
    cvNamedWindow("DstImage",CV_WINDOW_AUTOSIZE);  
    // 显示图像pSrc  
    cvShowImage("SrcImage",pSrcImg);  
    cvShowImage("DstImage",pDstImg);  
    cvWaitKey(0);  
    cvDestroyAllWindows();  
    cvReleaseImage(&pSrcImg);  
    cvReleaseImage(&pDstImg); 
}

void Resize_image_3( const unsigned char *p_srcimg_data, const int n_srcimg_widthstep, unsigned char *p_dstimg_data, const int n_dstimg_widthstep, \
					int n_srcimg_w, int n_srcimg_h,int n_dstimg_w, int n_dstimg_h, int n_channels )
{
	int n_w0 = n_srcimg_w;
	int n_h0 = n_srcimg_h;

	int n_w1 = n_dstimg_w;
	int n_h1 = n_dstimg_h;

	if( NULL == p_srcimg_data )
	{
		printf("input image is null!\n");
		assert(p_srcimg_data);
		return;
	}

	float f_scale_factor_w = (float)n_w1/(float)n_w0;
	float f_scale_factor_h = (float)n_h1/(float)n_h0;

	float f_x0 = 0, f_y0 = 0;
	int n_x1 = 0;
	int n_x2 = 0;
	int n_y1 = 0;
	int n_y2 = 0;

	float f_y1 = 0;
	float f_y2 = 0;

	int *pn_x1 = (int *)calloc(n_w1,sizeof(int));
	int *pn_x2 = (int *)calloc(n_w1,sizeof(int));
	float *pn_f_x1 = (float *)calloc(n_w1,sizeof(float));
	float *pn_f_x2 = (float *)calloc(n_w1,sizeof(float));

	float f_x1 = 0;
	float f_x2 = 0;

	float f_s1 = 0;
	float f_s2 = 0;
	float f_s3 = 0;
	float f_s4 = 0;

	float f_grayvalue;
	float f_R_val;
	float f_G_val;
	float f_B_val;

	int n_col,n_row;

	for(n_col = 0; n_col < n_w1; n_col++)
	{
		f_x0 = (float)n_col / f_scale_factor_w;
		pn_x1[n_col] = (int)(f_x0);
		pn_x2[n_col] = pn_x1[n_col] + 1;
		pn_f_x1[n_col] = f_x0 - pn_x1[n_col];
		pn_f_x2[n_col] = 1.0 - pn_f_x1[n_col];

		if( pn_x2[n_col] >= (n_w1-1)){ //
			pn_x2[n_col] = n_w1-1;
		}
	}

	for(n_row = 0; n_row < n_h1; n_row++)
	{
		f_y0 = (float)n_row / f_scale_factor_h;
		n_y1 = (int)(f_y0);
		n_y2 = n_y1 + 1;
		f_y1 = f_y0 - n_y1; //p
		f_y2 = 1.0 - f_y1; //1-p

		if(n_y2 >= (n_h1-1) ){
			n_y2 = n_h1-1;
		}

		for(n_col = 0; n_col < n_w1; n_col++)
		{
			n_x1 = pn_x1[n_col];
			n_x2 = pn_x2[n_col];
			f_x1 = pn_f_x1[n_col]; //q
			f_x2 = pn_f_x2[n_col]; //1-q

			f_s1 = f_x2 * f_y2; // 1-q * 1-p
			f_s2 = f_x1 * f_y2; // q * 1-p
			f_s3 = f_x1 * f_y1; // q * p
			f_s4 = f_x2 * f_y1; // 1-q * p

			if( 1 == n_channels ){

				f_grayvalue = p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x1] * f_s1;
				f_grayvalue += p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x2] * f_s2;
				f_grayvalue += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x1] * f_s4;
				f_grayvalue += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x2] * f_s3;

				p_dstimg_data[n_row * n_dstimg_widthstep + n_col] = (int)f_grayvalue;

			}else if( 3 == n_channels ){

				//Blue channel
				f_B_val = p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x1*3] * f_s1;
				f_B_val += p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x2*3] * f_s2;
				f_B_val += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x1*3] * f_s4;
				f_B_val += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x2*3] * f_s3;

				p_dstimg_data[n_row * n_dstimg_widthstep + n_col*3] = (int)f_B_val;
				
				//Green channel
				f_G_val = p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x1*3+1] * f_s1;
				f_G_val += p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x2*3+1] * f_s2;
				f_G_val += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x1*3+1] * f_s4;
				f_G_val += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x2*3+1] * f_s3;

				p_dstimg_data[n_row * n_dstimg_widthstep + n_col*3+1] = (int)f_G_val;

				//Red channel
				f_R_val = p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x1*3+2] * f_s1;
				f_R_val += p_srcimg_data[n_y1 * n_srcimg_widthstep + n_x2*3+2] * f_s2;
				f_R_val += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x1*3+2] * f_s4;
				f_R_val += p_srcimg_data[n_y2 * n_srcimg_widthstep + n_x2*3+2] * f_s3;

				p_dstimg_data[n_row * n_dstimg_widthstep + n_col*3+2] = (int)f_R_val;

			}else{

				free(pn_x1);
				free(pn_x2);
				free(pn_f_x1);
				free(pn_f_x2);
				return;

			}

			

			/*p_dstimg_data[n_row * n_w1 + n_col] = (int)(p_srcimg_data[n_y1 * n_w0 + n_x1] * f_s1 + p_srcimg_data[n_y2 * n_w0 + n_x1] * f_s2 + \
														  p_srcimg_data[n_y1 * n_w0 + n_x2] * f_s4 + p_srcimg_data[n_y2 * n_w0 + n_x2] * f_s3 );*/

		}
	}


	free(pn_x1);
	free(pn_x2);
	free(pn_f_x1);
	free(pn_f_x2);

	return;

}

void Resize_image_4( const unsigned char *p_srcimg_data, const int n_srcimg_widthstep, unsigned char *p_dstimg_data, const int n_dstimg_widthstep, \
					int n_srcimg_w, int n_srcimg_h, int n_dstimg_w, int n_dstimg_h, int n_channels )
{
	int srcImgHeight = n_srcimg_h;  
    int srcImgWidth = n_srcimg_w;  
    int srcImgWidthStep = n_srcimg_widthstep;  
    int dstImgWidthStep = n_dstimg_widthstep;  
  
    float heightRatio = (float)n_dstimg_h / (float)n_srcimg_h;  
    float widthRatio = (float)n_dstimg_w / (float)n_srcimg_w;  

	float fy;  
    int ny;  
    int nya1;  
    float p; 

	float fx ;  
    int nx ;  
    int nxa1;  
    float q ; 

	float f_gray_val;
	float f_r_val;
	float f_g_val;
	float f_b_val;

	int i,j;
  
    for(i = 0; i < n_dstimg_h; i++)  
    {  
        fy = (float)i/heightRatio;  
        ny = (int)fy;  
        nya1 = ny+1;  
        p = fy-ny; 

        if (nya1>=srcImgHeight) //最后一行  
        {  
            nya1 = srcImgHeight-1;  
        }  

        for(j = 0;j < n_dstimg_w; j++)  
        {  
            fx = (float)j/widthRatio;  
            nx = (int)fx;  
            nxa1 = nx+1;  
            q = fx - nx;  

            if (nxa1>=srcImgWidth) //该行最后一个元素  
            {  
                nxa1 = srcImgWidth - 1;  
            }  
  
			if(1 == n_channels){

				f_gray_val = (1-p)*(1-q)*p_srcimg_data[ny*srcImgWidthStep+nx];  
				f_gray_val += (1-p)*q*p_srcimg_data[ny*srcImgWidthStep+nxa1];  
				f_gray_val += p*(1-q)*p_srcimg_data[nya1*srcImgWidthStep+nx];  
			    f_gray_val += p*q*p_srcimg_data[nya1*srcImgWidthStep+nxa1];  
  
				p_dstimg_data[i*dstImgWidthStep+j]=(int)f_gray_val;

			}else if(3 == n_channels){

				f_b_val = (1-p)*(1-q)*p_srcimg_data[ny*srcImgWidthStep+nx*3];  
				f_b_val += (1-p)*q*p_srcimg_data[ny*srcImgWidthStep+nxa1*3];  
				f_b_val += p*(1-q)*p_srcimg_data[nya1*srcImgWidthStep+nx*3];  
				f_b_val += p*q*p_srcimg_data[nya1*srcImgWidthStep+nxa1*3];  
  
				p_dstimg_data[i*dstImgWidthStep+j*3]=(int)f_b_val;  
  
				f_g_val = (1-p)*(1-q)*p_srcimg_data[ny*srcImgWidthStep+nx*3+1];  
				f_g_val += (1-p)*q*p_srcimg_data[ny*srcImgWidthStep+nxa1*3+1];  
				f_g_val += p*(1-q)*p_srcimg_data[nya1*srcImgWidthStep+nx*3+1];  
				f_g_val += p*q*p_srcimg_data[nya1*srcImgWidthStep+nxa1*3+1];  
  
				p_dstimg_data[i*dstImgWidthStep+j*3+1]=(int)f_g_val;  
  
				f_r_val = (1-p)*(1-q)*p_srcimg_data[ny*srcImgWidthStep+nx*3+2];  
				f_r_val += (1-p)*q*p_srcimg_data[ny*srcImgWidthStep+nxa1*3+2];  
				f_r_val += p*(1-q)*p_srcimg_data[nya1*srcImgWidthStep+nx*3+2];  
				f_r_val += p*q*p_srcimg_data[nya1*srcImgWidthStep+nxa1*3+2];  
  
				p_dstimg_data[i*dstImgWidthStep+j*3+2]=(int)f_r_val; 
			}

        }  
    }  
}

void Image2double_test()
{
	cv::Mat src_mat = (cv::Mat_<unsigned short>(3,3) << 0,1,2,3,4,5,6,7,8 );
	cv::Mat_<float> dst_mat;

	src_mat.convertTo(dst_mat,CV_32F);
	std::cout << "-----------print correlation mat type---------" << std::endl;
	std::cout << "src_mat type : "<< src_mat.depth() << std::endl;
	std::cout << "dst_mat type : "<< dst_mat.depth() << std::endl;
	//std::cout << setiosflags(ios::scientific) << setprecision(8);
	std::cout << setiosflags(ios::fixed) << setprecision(8);
	std::cout << "src_mat = " << endl << " " << setprecision(8) << dst_mat << endl << endl;
	std::cout << "atan2f(22.00,12.00) = " << setprecision(8) << atan2f(22.00,12.00) << std::endl;

	//-----------------------------
	cv::Mat mat1=Mat(2,2,CV_32FC1);
	mat1.at<float>(0,0) = 1.0f;
	mat1.at<float>(0,1) = 2.0f;
	mat1.at<float>(1,0) = 3.0f;
	mat1.at<float>(1,1) = 4.0f;
	// 对于这种小矩阵，还有更简单的赋值方式，找时间再改
	cout<<"Mat 1:"<<endl;
	cout<<mat1<<endl;
	normalize(mat1,mat1,1.0,0.0,NORM_MINMAX);
	cout<<"Mat 2:"<<endl;
	cout<<mat1<<endl;
	//-----------------------------

	system("pause");
}
