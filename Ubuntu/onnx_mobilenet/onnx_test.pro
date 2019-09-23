QT += core
QT -= gui

TARGET = onnx_test
CONFIG += console
CONFIG -= app_bundle
CONFIG += C++11

TEMPLATE = app

SOURCES += main.cpp

INCLUDEPATH += /usr/include \
               /usr/include/opencv \
               /usr/include/opencv2 \
               /media/usr523/000903F80002AA1E/cxy/1908/1909/onnxruntime/include/onnxruntime

LIBS += /usr/lib/x86_64-linux-gnu/libopencv_highgui.so \
        /usr/lib/x86_64-linux-gnu/libopencv_core.so    \
        /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so \
        /media/usr523/000903F80002AA1E/cxy/1908/1909/onnxruntime/build/Linux/Release/libonnxruntime.so.0.5.0
