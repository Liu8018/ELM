TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    elm_model.cpp \
    elm_in_elm_model.cpp \
    delm_model.cpp \
    functions.cpp

INCLUDEPATH += usr/include\
               usr/include/opencv \
               usr/include/opencv2
LIBS += /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so \
        /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
        /usr/lib/x86_64-linux-gnu/libopencv_core.so \
        /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so

HEADERS += \
    elm_model.h \
    elm_in_elm_model.h \
    functions.h \
    delm_model.h
