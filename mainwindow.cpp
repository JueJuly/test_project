#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    bin = "D:/QtProject/opencv_test/test_img" ;
    //mo_file_info = FileInfo(this);
   // mo_current_pro_dir = QDir(bin);
    ms_win_name = "test opencv";
    QAction *p_action_open = ui->menuBar->addAction(QObject::tr("Open_Image"));

    connect( p_action_open, SIGNAL( triggered() ), this, SLOT( open_picture() ) );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::open_picture(){

    mo_img_path = QFileDialog::getOpenFileName(this, tr("Open Image"),
        bin, tr("Images(*.jpg *.bmp *.png)"));
    if( mo_img_path.isEmpty() ){
        return ;
    }
    QString path = QDir::currentPath();//当前路径
    qDebug() << path;
    //setCurrent 设置当前路径

    cv::namedWindow( ms_win_name, WINDOW_AUTOSIZE );
    std::string str_img_path = mo_img_path.toStdString();

    mo_src_img = imread( str_img_path, 1 );
    cv::imshow( ms_win_name , mo_src_img );

    cv::waitKey(0);
    cv::destroyAllWindows();

}
