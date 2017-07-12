
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
        delete[]pre_pyr[i];  
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
  
        for (int j = max_pyramid_layer - 1; j >= 0; j--)  
        {  
            DBPoint curpoint;  
            curpoint.x = start[i].x / pow(2.0, j);  
            curpoint.y = start[i].y / pow(2.0, j);  

            double Xleft = curpoint.x - window_radius;  
            double Xright = curpoint.x + window_radius;  
            double Yleft = curpoint.y - window_radius;  
            double Yright = curpoint.y + window_radius;  
			
			//得到梯度矩阵
            double gradient[4] = { 0 };  
            int cnt = 0;  
            for (double xx = Xleft; xx < Xright + 0.01; xx += 1.0)  
                for (double yy = Yleft; yy < Yright + 0.01; yy += 1.0)  
                {  
                    assert(xx < 1000 && yy < 1000 && xx >= 0 && yy >= 0);  

                    double derivativeX = get_subpixel(frame_pre[j],height[j], width[j], DBPoint(xx + 1.0, yy)) - \
						                   get_subpixel(frame_pre[j], height[j],  width[j], DBPoint(xx - 1.0, yy));  
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
                        assert(xx < 1000 && yy < 1000 && xx >= 0 && yy >= 0);  
                        double nextX = xx + g[0] + opticalflow[0];  
                        double nextY = yy + g[1] + opticalflow[1];  
                        assert(nextX < 1000 && nextY < 1000 && nextX >= 0 && nextY >= 0);  
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
            exit(0);  
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

int opticalFlowTest()
{
	std::string currentFramePath = "./opticalFlowTestData/secondFram.jpg";
	std::string nextFramePath = "./opticalFlowTestData/3thFrame.jpg";

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