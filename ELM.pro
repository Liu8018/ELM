TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    elm_model.cpp \
    elm_in_elm_model.cpp \
    functions.cpp

LIBS += /usr/lib64/libopencv_core.so \ 
        /usr/lib64/libopencv_imgproc.so \
        /usr/lib64/libopencv_highgui.so \
        /usr/lib64/libopencv_imgcodecs.so

HEADERS += \
    elm_model.h \
    elm_in_elm_model.h \
    functions.h
