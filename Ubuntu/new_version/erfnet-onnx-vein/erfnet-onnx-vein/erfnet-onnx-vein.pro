QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
TEMPLATE = app
# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp

INCLUDEPATH +=/usr/include/opencv\
              /usr/include/opencv2 \
              #/home/tzj/onnxruntime/include/onnxruntime \
              /home/tzj/onnxruntime/include/onnxruntime/core/session \
              /home/tzj/onnxruntime/include/onnxruntime/core/providers/cuda


LIBS +=   /usr/lib/libopencv_highgui.so \
          /usr/lib/libopencv_core.so    \
          /usr/lib/libopencv_imgproc.so \
          /usr/lib/libopencv_videoio.so \
          /home/tzj/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so.1.2.0 \
          /home/tzj/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so \




QMAKE_CXXFLAGS += -std=c++11 -g
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui





