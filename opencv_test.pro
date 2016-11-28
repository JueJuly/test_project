#-------------------------------------------------
#
# Project created by QtCreator 2016-11-21T21:40:53
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = opencv_test
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    fileinfo.cpp \
    fileview.cpp

HEADERS  += mainwindow.h \
    fileview.h

FORMS    += mainwindow.ui

INCLUDEPATH += ..\opencv\include

win32:CONFIG(debug, debug|release): {
    LIBS += -L..\opencv\lib \
    -lopencv_core2412d -lopencv_imgproc2412d -lopencv_highgui2412d -lopencv_ml2412d -lopencv_video2412d -lopencv_features2d2412d -lopencv_calib3d2412d \
    -lopencv_objdetect2412d -lopencv_contrib2412d -lopencv_legacy2412d -lopencv_flann2412d -lopencv_nonfree2412d

    #LIBS += -L..\src\detect -llaser_pt_detect_d
}
else:win32:CONFIG(release, debug|release): {
    LIBS += -L..\opencv\lib \
    -lopencv_core2412 -lopencv_imgproc2412 -lopencv_highgui2412 -lopencv_ml2412 -lopencv_video2412 -lopencv_features2d2412 -lopencv_calib3d2412 \
    -lopencv_objdetect2412 -lopencv_contrib2412 -lopencv_legacy2412 -lopencv_flann2412 -lopencv_nonfree2412
   #LIBS += -L..\src\detect -llaser_pt_detect_r
    #LIBS += -L..\src\hikvesion -lPlayCtrl -lHCNetSDK # hikvesion
}
