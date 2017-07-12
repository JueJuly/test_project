
#include "LucasKanadeTracker.h"  
   
using namespace std;  

LucasKanadeTracker::LucasKanadeTracker(const int windowRadius, bool usePyr)  
    :window_radius(windowRadius), isusepyramid(usePyr)  
{  
	
}  
  
  
LucasKanadeTracker::~LucasKanadeTracker()  
{  
    for (int i = 0; i < max_pyramid_layer; i++)  
    {  
        if (pre_pyr[i])  
            delete[]pre_pyr[i];  
        if (next_pyr[i])  
            delete[]next_pyr[i];  
    }  

    delete[]pre_pyr;  
    delete[]next_pyr;  

    if (height)  
        delete[]height;  
    if (width)  
        delete[]width;  
}  
  
  
void LucasKanadeTracker::lowpass_filter(BYTE *&src, const int H, const int W, BYTE *&smoothed)  
{  
    //tackle with border  
    for (int i = 0; i < H; i++)  
    {  
        smoothed[i*W] = src[i*W];  
        smoothed[i*W + W - 1] = src[i*W + W - 1];  
    }  
    for (int i = 0; i < W; i++)  
    {  
        smoothed[i] = src[i];  
        smoothed[(H - 1)*W + i] = src[(H - 1)*W + i];  
    }  
  
    for (int i = 1; i < H - 1; i++)  
        for (int j = 1; j < W - 1; j++)  
        {  
            double re = 0;  
            re += src[i*W + j] * 0.25;  
            re += src[(i - 1)*W + j] * 0.125;  
            re += src[i*W + j + 1] * 0.125;  
            re += src[i*W + j - 1] * 0.125;  
            re += src[(i + 1)*W + j] * 0.125;  
            re += src[(i - 1)*W + j - 1] * 0.0625;  
            re += src[(i + 1)*W + j - 1] * 0.0625;  
            re += src[(i - 1)*W + j + 1] * 0.0625;  
            re += src[(i + 1)*W + j + 1] * 0.0625;  
            smoothed[i*W + j] = BYTE(re);  
        }  
    delete[]src;  
    src = smoothed;  
}  
  
  
void LucasKanadeTracker::get_info(const int nh, const int nw)  
{  
    original_imgH = nh;  
    original_imgW = nw;  
    if (isusepyramid)  
        get_max_pyramid_layer();  
    else  
        max_pyramid_layer = 1;  

    pre_pyr = new BYTE*[max_pyramid_layer];  
    next_pyr = new BYTE*[max_pyramid_layer];  
    height = new int[max_pyramid_layer];  
    width = new int[max_pyramid_layer];  
    height[0] = nh;  
    width[0] = nw;  
}  
void LucasKanadeTracker::get_target(POINT *target, int n)  
{  
    this->target = new DBPoint[n];  
    endin = new DBPoint[n];  
    for (int i = 0; i < n; i++)  
    {  
        this->target[i].x = target[i].x;  
        this->target[i].y = target[i].y;  
    }  
    numofpoint = n;  
}  
  
BYTE *&LucasKanadeTracker::get_pyramid(int th)  
{  
    return pre_pyr[th];  
}  
POINT LucasKanadeTracker::get_result()  
{  
    POINT pp;  
    pp.x = target[0].x;  
    pp.y = target[0].y;  
    return pp;  
}  
void LucasKanadeTracker::get_pre_frame(BYTE *&gray)//use only at the beginning  
{  
    pre_pyr[0] = gray;  
    build_pyramid(pre_pyr);  
    //save_gray("1.bmp", pre_pyr[1], height[1], width[1]);  
}  
  
void  LucasKanadeTracker::discard_pre_frame()  
{  
    //we don't new memory for original data,so we don't delete it here  
    for (int i = 0; i < max_pyramid_layer; i++)  
	{
        //delete[]pre_pyr[i];
		pre_pyr[i] = NULL;
	}

}  

//set the next frame as pre_frame,must dicard pre_pyr in advance  
void  LucasKanadeTracker::get_pre_frame()  
{  
    for (int i = 0; i < max_pyramid_layer; i++)  
        pre_pyr[i] = next_pyr[i];  
}  
//use every time,must after using get_pre_frame(BYTE**pyr)  
void  LucasKanadeTracker::get_next_frame(BYTE*&gray)  
{  
    next_pyr[0] = gray;  
    build_pyramid(next_pyr);  
    //save_gray("1.bmp", next_pyr[1], height[1], width[1]);  
}  
  
//金字塔建立
void LucasKanadeTracker::pyramid_down( BYTE *& src_gray_data,  
    const int src_h, const int src_w, BYTE *& dst, int &dst_h, int &dst_w)  
{  
    dst_h = src_h / 2;  
    dst_w = src_w / 2;  
    int ii = height[1];  
    int hh = width[1];  
    assert(dst_w > 3 && dst_h > 3);  
    //BYTE*smoothed = new BYTE[src_h*src_w];  
    dst = new BYTE[dst_h*dst_w];  
    //lowpass_filter(src_gray_data, src_h, src_w,smoothed);  
    for (int i = 0; i < dst_h - 1; i++)  
	{
        for (int j = 0; j < dst_w - 1; j++)  
        {  
            int srcY = 2 * i + 1;  
            int srcX = 2 * j + 1;  
            double re = src_gray_data[srcY*src_w + srcX] * 0.25;  
            re += src_gray_data[(srcY - 1)*src_w + srcX] * 0.125;  
            re += src_gray_data[(srcY + 1)*src_w + srcX] * 0.125;  
            re += src_gray_data[srcY*src_w + srcX - 1] * 0.125;  
            re += src_gray_data[srcY*src_w + srcX + 1] * 0.125;  
            re += src_gray_data[(srcY - 1)*src_w + srcX + 1] * 0.0625;  
            re += src_gray_data[(srcY - 1)*src_w + srcX - 1] * 0.0625;  
            re += src_gray_data[(srcY + 1)*src_w + srcX - 1] * 0.0625;  
            re += src_gray_data[(srcY + 1)*src_w + srcX + 1] * 0.0625;  
            dst[i*dst_w + j] = re;  
        } 
	}
		
	//使用倒数第二列数据填充最后一列
    for (int i = 0; i < dst_h; i++)  
        dst[i*dst_w + dst_w - 1] = dst[i*dst_w + dst_w - 2]; 
	//使用倒数第二行数据填充最后一行
    for (int i = 0; i < dst_w; i++)  
        dst[(dst_h - 1)*dst_w + i] = dst[(dst_h - 2)*dst_w + i];  
}  
  
//bilinear interplotation  
double LucasKanadeTracker::get_subpixel(BYTE*&src, int h, int w, const DBPoint& point)  
{  
    int floorX = floor(point.x);  
    int floorY = floor(point.y);  
  
    double fractX = point.x - floorX;  
    double fractY = point.y - floorY;  
  
    return ((1.0 - fractX) * (1.0 - fractY) * src[floorX + w* floorY])   
               + (fractX * (1.0 - fractY) * src[floorX + 1 + floorY*w])   
               + ((1.0 - fractX) * fractY * src[floorX + (floorY + 1)*w])  
               + (fractX * fractY * src[floorX + 1 + (floorY + 1)*w]);  
}  
  
  
void LucasKanadeTracker::get_max_pyramid_layer()  
{  
    int layer = 0;  
    int windowsize = 2 * window_radius + 1;  
    int temp = original_imgH > original_imgW ? original_imgW : original_imgH;  
    if (temp >= ((1 << 4) * 2 * windowsize))  
    {  
        max_pyramid_layer = 5;  
        return;  
    }  
    temp = double(temp) / 2;  
    while (temp >= 2 * windowsize)  
    {  
        layer++;  
        temp = double(temp) / 2;  
    }  
    max_pyramid_layer = layer;  
}  
  
void LucasKanadeTracker::build_pyramid(BYTE **&pyramid)  
{  
    for (int i = 1; i < max_pyramid_layer; i++)  
    {  
        pyramid_down(pyramid[i - 1], height[i - 1],  
            width[i - 1], pyramid[i], height[i], width[i]);  
    }  
}  
  
void LucasKanadeTracker::run_single_frame()  
{  
    char*state = NULL;  
    lucaskanade(pre_pyr, next_pyr, target, endin, numofpoint, state);  
    for (int i = 0; i < numofpoint; i++)  
    {  
        target[i].x = endin[i].x;  
        target[i].y = endin[i].y;  
    }  
  
} 

void LucasKanadeTracker::run( POINT *&resultPt, int &nNum )
{
	char *state = NULL;  
    lucaskanade(pre_pyr, next_pyr, target, endin, numofpoint, state);  

#if 0	
	//以下是直接将计算出来的匹配点赋值给原来的目标点，
	//这样也把错误的点传过去
    for (int i = 0; i < numofpoint; i++)  
    {  
        target[i].x = endin[i].x;  
        target[i].y = endin[i].y; 

		resultPt[i].x = endin[i].x;
		resultPt[i].y = endin[i].y;
		printf("resultPt[%d]=%d,%d\n",i,resultPt[i].x,resultPt[i].y);
		nNum++;
    }
#endif

#if 1 //改进的方法1，对计算出来的结果点进行修正，
	  //具体做法：找出结果点集中与原目标点集中距离最小的点作为修正的点
	DBPoint minPt;
	minPt.x = 99999;
	minPt.y = 99999;

	//找最小距离点坐标
	for (int i = 0; i < numofpoint; i++)  
    {  
		if( ABS(target[i].x-endin[i].x) < minPt.x )
			minPt.x = endin[i].x;

		if( ABS(target[i].y-endin[i].y) < minPt.y )
			minPt.y = endin[i].y;

    }
	printf("minPt=%d,%d\n",minPt.x,minPt.y);
	//对点进行修正
	for (int i = 0; i < numofpoint; i++)  
    {  
		if( 0 == i )
		{
			if( endin[i].x < 0 || endin[i].x >= get_pyrW(0) )
			{
				//target[i].x = minPt.x;
				endin[i].x = minPt.x;
			}

			if( endin[i].y < 0 || endin[i].y >= get_pyrH(0) )
			{
				//target[i].y = minPt.y;
				endin[i].y = minPt.y;
			}
		}
		else
		{
			if( endin[i].x < 0 || endin[i].x >= get_pyrW(0) )
			{
				//target[i].x = minPt.x;
				endin[i].x = endin[i-1].x;
			}

			if( endin[i].y < 0 || endin[i].y >= get_pyrH(0) )
			{
				//target[i].y = minPt.y;
				endin[i].y = endin[i-1].y;
			}
		}
		target[i].x = endin[i].x;  
        target[i].y = endin[i].y;

		resultPt[i].x = endin[i].x;
		resultPt[i].y = endin[i].y;
		//printf("resultPt[%d]=%d,%d\n",i,resultPt[i].x,resultPt[i].y);
		nNum++;

		printf("target[%d]=%f,%f\n",i,target[i].x,target[i].y);
		printf("endin[%d]=%f,%f\n",i,endin[i].x,endin[i].y);
    }
#endif
}

POINT *LucasKanadeTracker::get_result( int &nNum )
{
	POINT *resultPt = new POINT[numofpoint];

	for (int i = 0; i < numofpoint; i++)  
    {   
		resultPt[i].x = endin[i].x;
		resultPt[i].y = endin[i].y;
		nNum++;
    }

	return resultPt;
}
  
void LucasKanadeTracker::lucaskanade( BYTE **&frame_pre, BYTE **&frame_cur,  
    DBPoint*& start, DBPoint*& finish, unsigned int point_nums, char*state )  
{  
    double*derivativeXs = new double[(2 * window_radius + 1)*(2 * window_radius + 1)];  
    double*derivativeYs = new double[(2 * window_radius + 1)*(2 * window_radius + 1)];  

    for (int i = 0; i < point_nums; i++)  
    {  
        double g[2] = { 0 };  
        double finalopticalflow[2] = { 0 };  
  
        memset(derivativeXs, 0, sizeof(double)* (2 * window_radius + 1)*(2 * window_radius + 1));  
  
        memset(derivativeYs, 0, sizeof(double)* (2 * window_radius + 1)*(2 * window_radius + 1));

		//printf("point_nums:%d\n",i);
  
        for (int j = max_pyramid_layer - 1; j >= 0; j--)  
        {  
            DBPoint curpoint;  
            curpoint.x = start[i].x / pow(2.0,j);  
            curpoint.y = start[i].y / pow(2.0,j); 
			//printf("layer_num:%d\n",j);
			
            double Xleft = curpoint.x - window_radius;  
            double Xright = curpoint.x + window_radius;  
            double Yleft = curpoint.y - window_radius;  
            double Yright = curpoint.y + window_radius;  
			//printf("left:%0.4f,right:%0.4f,top:%0.4f,down:%0.4f\n",Xleft,Xright,Yleft,Yright);
			//得到梯度矩阵
            double gradient[4] = { 0 };  
            int cnt = 0;  
            for (double xx = Xleft; xx < Xright + 0.01; xx += 1.0)  
                for (double yy = Yleft; yy < Yright + 0.01; yy += 1.0)  
                {  
                   //assert(xx < 1000 && yy < 1000 && xx >= 0 && yy >= 0); 
				   xx = (xx < 1.0) ? 1.0 : xx;
				   yy = (yy < 1.0) ? 1.0 : yy;
					//xx = (xx < 0) ? 0 : ( (xx > width[j]) ? width[j] : xx );
					//yy = (yy < 0) ? 0 : ( (yy > height[j]) ? height[j] : yy );

                    double derivativeX = get_subpixel( frame_pre[j],height[j], width[j], DBPoint(xx + 1.0, yy) ) - \
						                   get_subpixel( frame_pre[j], height[j], width[j], DBPoint(xx - 1.0, yy) );  
                    derivativeX /= 2.0;  
  
                    double t1 = get_subpixel(frame_pre[j], height[j], width[j], DBPoint(xx, yy + 1.0));  
                    double t2 = get_subpixel(frame_pre[j], height[j], width[j], DBPoint(xx, yy - 1.0));  

                    double derivativeY = (t1 - t2) / 2.0;  
  
                    derivativeXs[cnt] = derivativeX;  
                    derivativeYs[cnt++] = derivativeY;  
                    gradient[0] += derivativeX * derivativeX;  
                    gradient[1] += derivativeX * derivativeY;  
                    gradient[2] += derivativeX * derivativeY;  
                    gradient[3] += derivativeY * derivativeY;  
                }  

            double gradient_inv[4] = { 0 };  

            ContraryMatrix(gradient, gradient_inv, 2);  
  
            double opticalflow[2] = { 0 };  
            int max_iter = 50;  
            double opticalflow_residual = 1;  
            int iteration = 0;  

            while ( iteration < max_iter && opticalflow_residual > 0.00001)  
            {  
                iteration++;  
                double mismatch[2] = { 0 };  
                cnt = 0;  
                for (double xx = Xleft; xx < Xright + 0.001; xx += 1.0)  
                    for (double yy = Yleft; yy < Yright + 0.001; yy += 1.0)  
                    {  
                        //assert(xx < 2000 && yy < 2000 && xx >= 0 && yy >= 0);
						/*xx = (xx < 0) ? 0 : ( (xx > width[j]) ? width[j] : xx );
						yy = (yy < 0) ? 0 : ( (yy > height[j]) ? height[j] : yy );*/
						xx = (xx < 0) ? 0 : xx;
						yy = (yy < 0) ? 0 : yy;
                        double nextX = xx + g[0] + opticalflow[0];  
                        double nextY = yy + g[1] + opticalflow[1];

                        //assert(nextX < 2000 && nextY < 2000 && nextX >= 0 && nextY >= 0);  

						/*nextX = (nextX < 0) ? 0 : ( (nextX > width[j]) ? width[j] : nextX );
						nextY = (nextY < 0) ? 0 : ( (nextY > height[j]) ? height[j] : nextY );*/
						nextX = (nextX < 0) ? 0 : nextX;
						nextY = (nextY < 0) ? 0 : nextY;
                        double pixelDifference = (get_subpixel(frame_pre[j],height[j], width[j], DBPoint(xx, yy)) - \
													get_subpixel(frame_cur[j], height[j],width[j], DBPoint(nextX, nextY)));  
                        mismatch[0] += pixelDifference*derivativeXs[cnt];  
                        mismatch[1] += pixelDifference*derivativeYs[cnt++];  
                    }  
                double temp_of[2];  
                matrixMul(gradient_inv, 2, 2, mismatch, 2, 1, temp_of);  
                opticalflow[0] += temp_of[0];  
                opticalflow[1] += temp_of[1];  
                opticalflow_residual = abs(temp_of[0]) + abs(temp_of[1]);  
            }  
            if (j == 0)  
            {  
                finalopticalflow[0] = opticalflow[0];  
                finalopticalflow[1] = opticalflow[1];  
            }  
            else  
            {  
                g[0] = 2 * (g[0] + opticalflow[0]);  
                g[1] = 2 * (g[1] + opticalflow[1]);  
            }  
        }  
        finalopticalflow[0] += g[0];  
        finalopticalflow[1] += g[1];  
        finish[i].x = start[i].x + finalopticalflow[0];  
        finish[i].y = start[i].y + finalopticalflow[1];  
    }  
    delete[]derivativeXs, derivativeYs;  
}  
  
//matrix inverse  
void LucasKanadeTracker::ContraryMatrix(double *pMatrix, double * _pMatrix, int dim)  
{  
    double *tMatrix = new double[2*dim*dim];  
    for (int i = 0; i < dim; i++){  
        for (int j = 0; j < dim; j++)  
            tMatrix[i*dim * 2 + j] = pMatrix[i*dim + j];  
    }  
    for (int i = 0; i < dim; i++){  
        for (int j = dim; j < dim * 2; j++)  
            tMatrix[i*dim * 2 + j] = 0.0;  
        tMatrix[i*dim * 2 + dim + i] = 1.0;  
    }  
    //Initialization over!     
    for (int i = 0; i < dim; i++)//Process Cols     
    {  
        double base = tMatrix[i*dim * 2 + i];  
        if (fabs(base) < 1E-300){  
            // AfxMessageBox("求逆矩阵过程中被零除，无法求解!" );  
            //_ASSERTE(-1);//throw exception 
			assert(-1);
            //exit(0);  
			//return;
        }  
        for (int j = 0; j < dim; j++)//row     
        {  
            if (j == i) continue;  
            double times = tMatrix[j*dim * 2 + i] / base;  
            for (int k = 0; k < dim * 2; k++)//col     
            {  
                tMatrix[j*dim * 2 + k] = tMatrix[j*dim * 2 + k] - times*tMatrix[i*dim * 2 + k];  
            }  
        }  
        for (int k = 0; k < dim * 2; k++){  
            tMatrix[i*dim * 2 + k] /= base;  
        }  
    }  
    for (int i = 0; i < dim; i++)  
    {  
        for (int j = 0; j < dim; j++)  
            _pMatrix[i*dim + j] = tMatrix[i*dim * 2 + j + dim];  
    }  
    delete[] tMatrix;  
}  
  
bool LucasKanadeTracker::matrixMul(double *src1, int height1, int width1, double *src2, int height2, int width2, double *dst)  
{  
    int i, j, k;  
    double sum = 0;  
    double *first = src1;  
    double *second = src2;  
    double *dest = dst;  
    int Step1 = width1;  
    int Step2 = width2;  
  
    if (src1 == NULL || src2 == NULL || dest == NULL || height2 != width1)  
        return false;  
  
    for (j = 0; j < height1; j++)  
    {  
        for (i = 0; i < width2; i++)  
        {  
            sum = 0;  
            second = src2 + i;  
            for (k = 0; k < width1; k++)  
            {  
                sum += first[k] * (*second);  
                second += Step2;  
            }  
            dest[i] = sum;  
        }  
        first += Step1;  
        dest += Step2;  
    }  
    return true;  
}  

void printLog( const char *logInfo )
{
	int nLength ;
	time_t rawTime;
	struct tm *timeInfo;
	char strTime[500];
	char strTemp[2000];
	FILE *fp = NULL;
	nLength = (int)strlen(logInfo);

	if( nLength > 0 )
	{
		time(&rawTime);
		timeInfo = localtime(&rawTime);
		sprintf(strTime,"\nThe current date/time is: %s", asctime(timeInfo));
		fp = fopen("AlgoFindCorner.log","at+");
		fwrite(strTime,(int)(strlen(strTime)),1,fp);
		fwrite("\t",1,1,fp);
		fwrite(logInfo,nLength,1,fp);
		fflush(fp);
		fclose(fp);
	}
	

}

int run_video()
{
	//std::string videoPath = "./opticalFlowTestData/bike.avi";
	std::string videoPath = "./opticalFlowTestData/1.mp4";
	cv::VideoCapture capture(videoPath);
	std::size_t Pos = videoPath.find_last_of("/");
	std::string fileName = videoPath.substr(Pos+1);
	long startFrame = 460;
	long frameSeq = startFrame;

	long totalFrameNum = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cv::Mat frameImg;
	cv::Mat frameGrayImg;
	cv::Mat smallImg;
	std::string writeFilePath ;
	char writeFileName[500];
	stringstream ss;
	//角点区域
	int nLeft = 21;
	int nRight = 42;
	int nTop = 195;
	int nDown = 209;
	POINT *testPt = new POINT[(nRight-nLeft)*(nDown-nTop)];
	POINT *DisPt = new POINT[(nRight-nLeft)*(nDown-nTop)];
	int nPtNum = 0;
	int nHight = 360;
	int nWidth = 380;
	BYTE *currFrameData = new BYTE[nHight*nWidth];
	BYTE *nextFrameData = new BYTE[nHight*nWidth];
	
	//BYTE *copydate = currentGrayCopy.data;

	std::cout << "fileName:" << fileName << std::endl;
	std::cout << "the total frame number：" << totalFrameNum << std::endl;

	if( !capture.isOpened() )
	{
		std::cout << "open video fail!" << std::endl;
		fprintf(stderr,"Return Error Code：%d\n",VIDEO_OPEN_FAIL);
		return VIDEO_OPEN_FAIL;
	}

	for(int h = nTop; h < nDown; h++ )
	{
		for( int w = nLeft; w < nRight; w++ )
		{
			testPt[(h-nTop)*(nRight-nLeft)+(w-nLeft)].x = w;
			testPt[(h-nTop)*(nRight-nLeft)+(w-nLeft)].y = h;
		}
	}

	/*for(int i = 0; i < (nRight-nLeft)*(nDown-nTop); i++ )
	{
		printf("testPt[%d]:%d,%d\n",i,testPt[i].x,testPt[i].y );	
	}*/

	//bike.avi特征点集
	/*testPt[0].x = 84;
	testPt[0].y = 113;
	testPt[1].x = 82;
	testPt[1].y = 115;
	testPt[2].x = 88;
	testPt[2].y = 123;
	testPt[3].x = 80;
	testPt[3].y = 123;
	testPt[4].x = 84;
	testPt[4].y = 122;
	testPt[5].x = 83;
	testPt[5].y = 126;
	testPt[6].x = 80;
	testPt[6].y = 125;
	testPt[7].x = 89;
	testPt[7].y = 127;
	testPt[8].x = 89;
	testPt[8].y = 129;
	testPt[9].x = 83;
	testPt[9].y = 129;
	testPt[10].x = 81;
	testPt[10].y = 130;
	testPt[11].x = 88;
	testPt[11].y = 131;
	testPt[12].x = 92;
	testPt[12].y = 134;
	testPt[13].x = 91;
	testPt[13].y = 136;
	testPt[14].x = 86;
	testPt[14].y = 134;
	testPt[15].x = 81;
	testPt[15].y = 135;*/

	//read from the first frame of video
	capture.set( CV_CAP_PROP_POS_FRAMES,startFrame);
	LucasKanadeTracker lkTracker(7,1);
	//lkTracker.get_info(240, 320);
	lkTracker.get_info(nHight, nWidth);
	
	while(frameSeq < totalFrameNum)
	{
		capture.set( CV_CAP_PROP_POS_FRAMES,frameSeq);

		if(!capture.read(frameImg))
		{
			fprintf(stderr,"Return Error Code：%d\n",READ_FRAME_IMAGE_FAIL);
			break;
			//return READ_FRAME_IMAGE_FAIL;
		}

		cvtColor(frameImg(cv::Rect(1060,480,380,360)),frameGrayImg,CV_RGB2GRAY);
		 
		memcpy(currFrameData,frameGrayImg.data,sizeof(BYTE)*nHight*nWidth);

		if( 0 == (frameSeq - startFrame) )
		{
			lkTracker.get_target(testPt,250);
			lkTracker.get_pre_frame(currFrameData);
			lkTracker.get_next_frame(currFrameData);
			memcpy(nextFrameData,currFrameData,sizeof(BYTE)*nHight*nWidth);
		}
		else
		{
			nPtNum = 0;
			lkTracker.get_pre_frame(nextFrameData);
			lkTracker.get_next_frame(currFrameData);
			lkTracker.run(DisPt,nPtNum);
			memcpy(nextFrameData,currFrameData,sizeof(BYTE)*nHight*nWidth);
		}

		for( int i = 0; i < lkTracker.max_pyramid_layer; i++ )
		{
			cv::Mat pyramidImg(lkTracker.get_pyrH(i),lkTracker.get_pyrW(i),CV_8UC1,(uchar*)lkTracker.pre_pyr[i]);
			cv::Mat next_pyrImg(lkTracker.get_pyrH(i),lkTracker.get_pyrW(i),CV_8UC1,(uchar*)lkTracker.next_pyr[i]);
			cv::Mat subMatImg = next_pyrImg - pyramidImg;

			/*for( int r = 0; r < subMatImg.rows; r++ )
			{
				for( int c = 0; c < subMatImg.cols; c++ )
				{
					if( subMatImg.at<uchar>(r,c) > 255 )
						subMatImg.at<uchar>(r,c) = 255;

					if( subMatImg.at<uchar>(r,c) < 0 )
						subMatImg.at<uchar>(r,c) = 0;
				}
			}*/

			imshow("next_pyrImg",next_pyrImg);
			imshow("pyramidImg",pyramidImg);
			imshow("subImg",subMatImg);
			waitKey(10);
		}

		printf("frameSeq=%d\n",frameSeq);
		for( int i = 0; i < nPtNum; i++ )
		{
			circle(frameImg(cv::Rect(1060,480,380,360)),cv::Point(DisPt[i].x,DisPt[i].y),2,CV_RGB(255,0,0),1);
			//printf("DisPt[%d]=%0.4f,%0.4f\n",i,DisPt[i].x,DisPt[i].y);
		}
		imshow("LeftframeImg",frameImg(cv::Rect(1060,480,380,360)));
		//imshow("RightframeImg",frameImg(cv::Rect(720,0,720,480)));
		//imshow("LeftframeImg",frameImg(cv::Rect(720,480,720,480)));
		cv::waitKey(25);
		

		//将读出的帧图像保存
		//---------------------
		/*ss << "./opticalFlowTestData/1_ImgSet/Right" << frameSeq << ".jpg" ;
		ss >> writeFilePath;
		ss.clear();
		ss.str("");
		frameGrayImg(cv::Rect(720,0,720,480)).copyTo(smallImg);
		imwrite(writeFilePath,smallImg);

		ss << "./opticalFlowTestData/1_ImgSet/Left" << frameSeq << ".jpg" ;
		ss >> writeFilePath;
		ss.clear();
		ss.str("");
		frameGrayImg(cv::Rect(720,480,720,480)).copyTo(smallImg);
		imwrite(writeFilePath,smallImg);*/
		//-------------------------
		frameSeq += 1;
	}

	delete []testPt;
	delete []DisPt;

	return RETURN_SUCCESS;
}

int opticalFlowTest()
{
	std::string currentFramePath = "./opticalFlowTestData/secondFram.jpg";
	std::string nextFramePath = "./opticalFlowTestData/3thFrame.jpg";
	const char *substring = strrchr(currentFramePath.c_str(),'/');
	std::string sunString = substring;
	std::cout << "substring:" << sunString << endl;
	std::size_t found = currentFramePath.find_last_of("/");
	std::cout << "path:" << currentFramePath.substr(0,found) << std::endl;
	std::cout << "file:" << currentFramePath.substr(found+1) << std::endl;

	cv::Mat currentMat = imread(currentFramePath,IMREAD_COLOR);
	cv::Mat nextMat = imread(nextFramePath,IMREAD_COLOR);

	cv::Mat currentGrayMat;
	cv::Mat nextGrayMat ;
	cvtColor( currentMat, currentGrayMat, CV_RGB2GRAY );
	cvtColor( nextMat, nextGrayMat, CV_RGB2GRAY );

	cv::Mat currentGrayCopy;
	currentGrayMat.copyTo(currentGrayCopy);

	BYTE *currData = currentGrayMat.data;
	BYTE *nextData = nextGrayMat.data;
	BYTE *copydate = currentGrayCopy.data;

	POINT *testPt = new POINT[10];

	testPt[0].x = 166;
	testPt[0].y = 125;

	testPt[1].x = 164;
	testPt[1].y = 132;

	testPt[2].x = 163;
	testPt[2].y = 141;

	testPt[3].x = 173;
	testPt[3].y = 126;

	testPt[4].x = 175;
	testPt[4].y = 140;

	testPt[5].x = 170;
	testPt[5].y = 126;

	testPt[6].x = 163;
	testPt[6].y = 128;

	/*testPt[0].x = 126;
	testPt[0].y = 120;

	testPt[1].x = 130;
	testPt[1].y = 126;

	testPt[2].x = 128;
	testPt[2].y = 130;

	testPt[3].x = 119;
	testPt[3].y = 126;

	testPt[4].x = 120;
	testPt[4].y = 119;

	testPt[5].x = 128;
	testPt[5].y = 134;

	testPt[6].x = 124;
	testPt[6].y = 130;*/

	for( int i = 0; i < 7; i++ )
	{
		circle(currentMat,cv::Point(testPt[i].x,testPt[i].y),2,CV_RGB(255,0,0),1);
	}
	imshow("currentFrame",currentMat);

	LucasKanadeTracker lktracker(7,1);
	lktracker.get_target(testPt,7);
	lktracker.get_info(240, 320); 
	lktracker.get_pre_frame(currData);
	lktracker.get_next_frame(nextData);
	lktracker.run_single_frame();
	
	for( int i = 0; i < lktracker.max_pyramid_layer; i++ )
	{
		cv::Mat pyramidImg(lktracker.get_pyrH(i),lktracker.get_pyrW(i),CV_8UC1,(uchar*)lktracker.pre_pyr[i]);
		imshow("pyramidImg",pyramidImg);
		waitKey(0);
	}

	for( int i = 0; i < 7; i++ )
	{
		circle(nextMat,cv::Point(lktracker.endin[i].x,lktracker.endin[i].y),2,CV_RGB(255,0,0),1);
	}
	imshow("nextMat",nextMat);

	waitKey(0);
	delete []testPt;
	
	//lktracker.~LucasKanadeTracker();

	return 0;

}