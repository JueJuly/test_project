#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtWidgets/QDialog>
#include <QString>
#include <QDebug>
#include <QObject>
#include <QPushButton>
#include <iostream>
#include <opencv.hpp>
#include <QMessageBox>
#include <QFileDialog>
#include <QPaintEvent>
#include <QImage>
#include <QTimer>
#include <QMutex>
#include <QPainter>
#include <QDateTime>
#include <QLabel>
#include <QMouseEvent>
#include <QEvent>

using namespace cv;
using namespace std;

const int WIDHT_OFFSET = 30;
const int HEIGHT_OFFSET = 30;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QMessageBox m_msgBox;
    cv::Mat m_srcImg;
    cv::Mat m_grayImg;
    cv::Mat m_smallImg;
    cv::Mat m_bigImg;
    cv::Mat m_resizeImg;

    QImage m_Image;
    //QPushButton *m_openImg;
    QAction *p_action_openImg;
    QAction *pActionGenerateData;
    QAction *pSavePtCoordAct;
    QLabel *m_imgLabel;
    QString m_filePath;
    QString m_fileName;
    QString mLabelBackgroundPath;
    QWidget *m_widget = NULL;
    QDialog *m_dialog = NULL;
    QLabel *m_label_1 = NULL;

    std::vector<cv::Point> ptVect;
    cv::Rect mRect;
/*
    void mousePressEvent(QMouseEvent *);
*/
    //void mouseMoveEvent(QMouseEvent *);

    //void myevent( QEvent *event);
    bool eventFilter(QObject *, QEvent *);

private slots:
    void m_open_Image();
    void m_generateData();
    void m_putText_img();
    void m_save_Ptcoord_data();
signals:
    void m_putText();

};

#endif // MAINWINDOW_H
