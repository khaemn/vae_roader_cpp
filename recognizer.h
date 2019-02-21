#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <QQuickItem>
#include <QRect>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable
#include <string>
#include <memory>

#ifdef _WIN32
#define OPENCV
#define GPU
#endif

// To use tracking - uncomment the following line. Tracking is supported only by OpenCV 3.x
//#define TRACK_OPTFLOW

//#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include\cuda_runtime.h"
//#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.1/lib/x64/cudart.lib")
//static std::shared_ptr<image_t> device_ptr(NULL, [](void *img) { cudaDeviceReset(); });

#define OPENCV
#include "yolo_v2_class.hpp"    // imported functions from DLL

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Recognizer : public QQuickItem
{
    Q_OBJECT
public:
    Recognizer();
    void init_default();

    Q_INVOKABLE void recognize();
    Q_INVOKABLE void setCamera(QQuickItem* ptr);

    std::vector<bbox_t> recognize(const cv::Mat& input, float thres=0.5);

    static const std::map<int,std::string> OBJ_NAMES;
signals:
    void recognized( int x_pos , int y_pos , int width , int heigth);

public slots:
    void scan();
private:
    QQuickItem* m_camera = nullptr;
    QSharedPointer<QQuickItemGrabResult> m_result;
    std::shared_ptr<Detector>  m_detector;
};

#endif // RECOGNIZER_H
