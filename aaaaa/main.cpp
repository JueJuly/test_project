#include "mainwindow.h"
#include <QApplication>
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

using namespace cv;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    //cv::waitKey(0);

    return a.exec();
}
