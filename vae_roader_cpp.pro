QT += quick
CONFIG += c++14 opencv

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += OPENCV
DEFINES += GPU

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    roaddetector.cpp \
    recognizer.cpp
RESOURCES += qml.qrc

LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
INCLUDEPATH += ../darknet/src/
INCLUDEPATH += ../darknet/include/
LIBS += ../darknet/libdarknet.so

QMAKE_LFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3

# QMAKE_CXXFLAGS +=-DSIMDPP_ARCH_X86_SSE4_1 -msse4 -msse3

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    roaddetector.h \
    recognizer.h


