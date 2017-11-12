#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <cstring>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ptVect.clear();
    //setFocus();
    //setMouseTracking ( true);
    this->setGeometry(450,450,800,600);
    //ui->setFixedSize();
    this->setFixedSize(800,600);
    setWindowTitle(tr("工具"));
    setWindowIcon(QIcon("Tool.ico"));
    m_imgLabel = new QLabel(this);
    m_imgLabel->setFocus();
    m_imgLabel->setText("test");
    //m_imgLabel->setStyleSheet("background-color:red");
    m_imgLabel->setGeometry(0,23,640,480);
    //m_imgLabel->installEventFilter(this);
    m_imgLabel->setMouseTracking(true);

    //m_Image = new QImage;
    p_action_openImg = ui->menuBar->addAction(QObject::tr("打开图像"));
    pActionGenerateData = ui->menuBar->addAction(QObject::tr("生成图像数据"));
    pSavePtCoordAct = ui->menuBar->addAction(QObject::tr("保存点坐标"));

    connect(p_action_openImg,SIGNAL(triggered()),this,SLOT(m_open_Image()));
    connect(pActionGenerateData,SIGNAL(triggered()),this,SLOT(m_generateData()));
    connect( this, SIGNAL(m_putText() ), this, SLOT( m_putText_img() ) );
    connect(pSavePtCoordAct,SIGNAL(triggered()),this,SLOT(m_save_Ptcoord_data()));

}

MainWindow::~MainWindow()
{
    delete ui;
    delete m_imgLabel;

}

void MainWindow::m_open_Image()
{
    //QString filename;
   // const std::string str;

    m_filePath = QFileDialog::getOpenFileName( this, tr("Open image"), "", tr("video(*.bmp *.jpg *.png)") );

    if( m_filePath.isEmpty() || m_filePath.isNull() )
    {
        m_msgBox.setText(tr("file path is error!"));
        m_msgBox.exec();
        return ;
    }
    else
    {
        qDebug() << m_filePath;
        //mLabelBackgroundPath = m_filePath;
        int first = m_filePath.lastIndexOf("/"); //从后面查找"/"位置
        QString title = m_filePath.right(m_filePath.length()-first-1); //从右边截取
        //qDebug() << title;
        int index = title.lastIndexOf(".");
        m_fileName = title.left(index);
        //qDebug() << m_fileName;

        const string str = m_filePath.toStdString();
        //cv::Mat tempImg;
        m_srcImg = cv::imread(str,IMREAD_COLOR);

        cv::cvtColor(m_srcImg,m_grayImg,CV_BGR2GRAY);
        //m_Image->load(filename);
        //cv::resize(tempImg,m_srcImg,cv::Size(640,480),0,0,CV_INTER_AREA);

    }

    emit m_putText();
}

void MainWindow::m_putText_img()
{
    //putText(m_srcImg,"test text",Point(20,20),CV_FONT_HERSHEY_PLAIN, 1.2, CV_RGB(0,255,0),2);
    m_Image = QImage(m_grayImg.data,m_srcImg.cols,m_srcImg.rows,QImage::Format_Grayscale8);
    m_bigImg = cv::Mat(m_srcImg.rows+2*HEIGHT_OFFSET,m_srcImg.cols+2*WIDHT_OFFSET,CV_8UC3);
    memset(m_bigImg.data,0,sizeof(uchar)*m_bigImg.rows*m_bigImg.cols*3);
    m_srcImg.copyTo(m_bigImg(cv::Rect(WIDHT_OFFSET,HEIGHT_OFFSET,m_srcImg.cols,m_srcImg.rows)));

    m_imgLabel->resize(m_Image.width(),m_Image.height());
    m_imgLabel->setPixmap(QPixmap::fromImage(m_Image));
   // QString tempStr;
    mLabelBackgroundPath = "QLabel{border-image: url(";
    mLabelBackgroundPath.append(m_filePath);
    mLabelBackgroundPath.append(");}");

    //m_imgLabel->setStyleSheet("QLabel{border-image: url(D:/QtProject/bbb/test_img/eg1.jpg);}");//图片在资源文件中
    m_imgLabel->setStyleSheet(mLabelBackgroundPath);
    m_imgLabel->installEventFilter(this);
    //cv::imshow("test",m_bigImg);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
}

void MainWindow::m_generateData()
{
    //qDebug() << m_filePath;
    char fileName[100];
    std::string tempFileName;
    std::string PtFileName;

    tempFileName = m_fileName.toStdString();
    PtFileName = tempFileName;
    tempFileName.append(".txt");
    PtFileName.append("PtCoord.txt");

    sprintf_s(fileName,"%s",tempFileName.c_str());

    //std::cout << fileName << std::endl;

    FILE *fp = fopen(fileName,"w");
    unsigned char *data = NULL;


    for(int h = 0; h < m_grayImg.rows; h++ )
    {
        data = m_grayImg.data + h * m_grayImg.cols;

        for( int w = 0; w < m_grayImg.cols; w++ )
        {
            if( !(w % 16))
            {
                fprintf_s(fp,"\n0x%02x,",data[w]);
            }
            else
            {
                fprintf_s(fp,"0x%02x,",data[w]);
            }
        }
    }

    fclose(fp);
    ptVect.clear();

    m_msgBox.setText(QString(tr("文件已经生成!\n文件名为:")).append(QString(fileName)));
    //m_msgBox.setText();
    m_msgBox.exec();

}

void MainWindow::m_save_Ptcoord_data()
{
    char fileName[100];
    std::string PtFileName;

    PtFileName = m_fileName.toStdString();
    PtFileName.append("PtCoord.txt");
    sprintf_s(fileName,"%s",PtFileName.c_str());

    //std::cout << fileName << std::endl;

    FILE *fp = fopen(fileName,"w");

    for( size_t i = 0; i < ptVect.size(); i++ )
    {
        if( i == 0)
            fprintf_s(fp,"%d,%d",ptVect[i].x,ptVect[i].y);
        else
            fprintf_s(fp,"\n%d,%d",ptVect[i].x,ptVect[i].y);
    }

    fclose(fp);
    ptVect.clear();

    m_msgBox.setText(QString(tr("点坐标已保存到文件!\n文件名为:")).append(QString(fileName)));
    m_msgBox.exec();
}

/*
void MainWindow::mousePressEvent(QMouseEvent *e)
{
     qDebug()<<"void MainWindow::mousePressEvent(QMouseEvent *)";
}
*/
//void MainWindow::mouseMoveEvent(QMouseEvent *e)
//{
//    e->accept();
//    qDebug()<<"void MainWindow::mouseMoveEvent(QMouseEvent *)";
//}

//void MainWindow::myevent( QEvent *e)
//{
//    if (e->type() == QEvent::MouseButtonPress)
//    {
//        QMouseEvent *event = static_cast<QMouseEvent*> (e);
//        qDebug() << QString("Press: %1, %2").arg(QString::number(event->x()), QString::number(event->y()));

//    }
//    else if (e->type() == QEvent::MouseButtonRelease)
//    {
//        QMouseEvent *event = static_cast<QMouseEvent*> (e);
//        qDebug() << QString("Release: %1, %2").arg(QString::number(event->x()), QString::number(event->y()));
//    }
//    else if (e->type() == QEvent::MouseMove)
//    {
//        QMouseEvent *event = static_cast<QMouseEvent*> (e);
//        qDebug() << QString("Move Point: %1, %2").arg(QString::number(event->pos().x()),QString::number(event->pos().y()));
//        qDebug() << QString("Move: %1, %2").arg(QString::number(event->x()), QString::number(event->y()));

//    }
//    else if (e->type() == QEvent::MouseMove)
//    {
//        //QMouseEvent *event = static_cast<QMouseEvent*> (e);
//        qDebug() << "---------------Cursor change !----------";

//    }
//}

bool MainWindow::eventFilter(QObject *target, QEvent *e)
{
    static bool bFlag = false;

   if(target == m_imgLabel)
   {
       if(e->type() == QEvent::Enter)
       {
           QMouseEvent *event = static_cast<QMouseEvent*> (e);
           //cv::namedWindow("局部放大图像");
           /*
           if( bFlag )
           {
               int w = (event->x() < 10) ? event->x() : (event->x()-10);
               int h = (event->y() < 10) ? event->y() : (event->y()-10);
               m_smallImg = m_srcImg(cv::Rect(w,h,20,20)).clone();
               cv::resize(m_smallImg,m_resizeImg,cv::Size(80,80),0,0,CV_INTER_AREA);
                qDebug() << "QEvent MouseMove!";
               QImage qImage = QImage(m_resizeImg.data,m_resizeImg.cols,m_resizeImg.rows,QImage::Format_RGB888);
               m_label_1->resize(qImage.width(),qImage.height());
               m_label_1->setPixmap(QPixmap::fromImage(qImage));
               m_dialog->show();
           }
           */


           return true;
       }
       else if(e->type() == QEvent::Leave)
       {
           qDebug() << "QEvent Leave!";
           //cv::destroyAllWindows();
           delete m_label_1;
           m_label_1 = NULL;
           delete m_dialog;
           m_dialog = NULL;
           delete m_widget;
           m_widget = NULL;
           bFlag = false;
           return true;
       }
       else if(e->type() == QEvent::MouseMove)
       {
           qDebug() << "QEvent MouseMove!";
           QMouseEvent *event = static_cast<QMouseEvent*> (e);
           //qDebug() << QString("Move Point: %1, %2").arg(QString::number(event->pos().x()),QString::number(event->pos().y()));
           //qDebug() << QString("Move: %1, %2").arg(QString::number(event->x()), QString::number(event->y()));

            if( bFlag )
            {
                mRect.x = event->x();
                mRect.y = event->y();
                mRect.width = WIDHT_OFFSET*2;
                mRect.height = HEIGHT_OFFSET*2;
                cv::Point Pt1;
                cv::Point Pt2;
                cv::Point Pt3;
                cv::Point Pt4;

                m_smallImg = m_bigImg(mRect).clone();
                //putText(m_smallImg,"+",Point(event->x()+WIDHT_OFFSET,event->y()+HEIGHT_OFFSET),CV_FONT_HERSHEY_PLAIN, 1.2, CV_RGB(255,0,0),2);

                cv::resize(m_smallImg,m_resizeImg,cv::Size(WIDHT_OFFSET*6,HEIGHT_OFFSET*6),0,0,CV_INTER_AREA);
                cv::cvtColor(m_resizeImg,m_resizeImg,CV_BGR2RGB);

                //画水平直线的两个点
                Pt1.x = m_resizeImg.cols/2-12;
                Pt2.x = m_resizeImg.cols/2+12;
                Pt1.y = m_resizeImg.rows/2;
                Pt2.y = m_resizeImg.rows/2;
                //画竖直直线的两个点
                Pt3.x = m_resizeImg.cols/2;
                Pt4.x = m_resizeImg.cols/2;
                Pt3.y = m_resizeImg.rows/2-12;
                Pt4.y = m_resizeImg.rows/2+12;
                cv::line(m_resizeImg,Pt1,Pt2,CV_RGB(0,0,255),1);
                cv::line(m_resizeImg,Pt3,Pt4,CV_RGB(0,0,255),1);
                //qDebug() << "QEvent MouseMove!";
                QImage qImage = QImage(m_resizeImg.data,m_resizeImg.cols,m_resizeImg.rows,QImage::Format_RGB888);
                m_label_1->resize(qImage.width(),qImage.height());
                m_label_1->setPixmap(QPixmap::fromImage(qImage));
                m_dialog->show();
            }

           return true;
       }
       else if(e->type() == QEvent::MouseButtonPress)
       {
           if(NULL == m_widget)
           {
               m_widget = new QWidget();
           }
           if( NULL == m_dialog)
           {
                m_dialog = new QDialog(m_widget);
           }
           if( NULL == m_label_1)
           {
               m_label_1 = new QLabel(m_dialog);
           }
            m_dialog->setGeometry(300,300,6*WIDHT_OFFSET,6*HEIGHT_OFFSET);
            m_dialog->setFixedSize(6*WIDHT_OFFSET,6*HEIGHT_OFFSET);
            bFlag = true;
           QMouseEvent *event = static_cast<QMouseEvent*> (e);
           if( event->button() == Qt::RightButton )
           {
               ptVect.push_back(cv::Point(event->x(),event->y()));
               qDebug() << QString("Move: %1, %2").arg(QString::number(event->x()), QString::number(event->y()));
           }
           else
           {
               qDebug() << "Mouse Event is not Right Buttion!";
           }

       }
   }

   return true;
}
