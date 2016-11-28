#ifndef MAINWINDOW_H
#define MAINWINDOW_H
//Qt
#include <QMainWindow>
#include <QFileDialog>
#include <QString>
#include <QDir>
#include <QDebug>
//C++
#include <string>
//opencv
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"



using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    cv::Mat mo_src_img;
    QString mo_img_path;
    String ms_win_name;
    QDir mo_current_pro_dir;
    QString bin;

    //FileInfo mo_file_info;

protected slots:
    void open_picture();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
