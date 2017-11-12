#-------------------------------------------------
#
# Project created by QtCreator 2016-01-20T21:04:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = aaaaa
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += d:\opencv\build\include
INCLUDEPATH += d:\opencv\build\include\opencv
INCLUDEPATH += d:\opencv\build\include\opencv2

LIBS += D:\opencv\compile\lib\libopencv_calib3d2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_contrib2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_core2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_features2d2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_flann2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_gpu2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_highgui2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_imgproc2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_legacy2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_ml2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_nonfree2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_objdetect2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_ocl2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_photo2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_stitching2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_superres2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_video2412.dll.a
LIBS += D:\opencv\compile\lib\libopencv_videostab2412.dll.a

#RESOURCES += \
#    icon.qrc
RC_FILE += icon.rc
