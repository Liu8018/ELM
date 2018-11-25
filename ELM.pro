TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    elm_model.cpp

LIBS += /usr/lib64/libopencv_core.so \      #核心部分
        /usr/lib64/libopencv_imgproc.so \
        /usr/lib64/libopencv_highgui.so \    #测试用
        /usr/lib64/libopencv_imgcodecs.so

HEADERS += \
    elm_model.h
