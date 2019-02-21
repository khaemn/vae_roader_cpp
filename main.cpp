#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QDateTime>
#include <QDebug>

#include <iostream>
#include <string>
#include <chrono>

#include <fdeep/fdeep.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "roaddetector.h"
#include "recognizer.h"

int main(int argc, char *argv[])
{
    using namespace cv;
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);

    RoadDetector detector;
    detector.load_model("../resources/mini_model_yolike_roader.json");

    Recognizer recognizer;

    //std::string filepath ("../resources/video/kiev/kiev2.mp4");
    //std::string filepath ("../resources/video/test/test-road-3.mp4");
    std::string filepath ("../resources/video/road9.mp4");

    cv::VideoCapture in_video = cv::VideoCapture(filepath);
    float framerate = in_video.get(CV_CAP_PROP_FRAME_COUNT);
    Size out_size(in_video.get(CV_CAP_PROP_FRAME_WIDTH), in_video.get(CV_CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer("../resources/out.avi",
                       CV_FOURCC('X', 'V', 'I', 'D'),
                       15.,
                       out_size,
                       true);
    if (!writer.isOpened()) {
        qDebug() << "Can not open video writer!";
    }

    cv::Mat in_frame;

    constexpr int frame_divider = 0;
    int frame_counter = 0;

    const Scalar road_outline_color( 50, 255, 50 );
    const Scalar red_color( 0, 0, 255 );
    const Scalar black_color( 0, 0, 0 );
    const Scalar white_color( 255, 255, 255 );

    const int road_outline_thickness = 3; // pixels

    while (in_video.read(in_frame) && in_video.isOpened()) {
        if (frame_counter++ > frame_divider) {
            frame_counter = 0;
        } else {
            continue;
        }

        auto start = std::chrono::system_clock::now();

        auto road_shape = detector.approx_road_shape(in_frame, 7.);

        if (!road_shape.empty()) {
            std::vector<std::vector<Point>> shapes { road_shape };
            drawContours( in_frame, shapes, 0, road_outline_color, road_outline_thickness);
        }

        // do some work
        std::vector<bbox_t> res = recognizer.recognize(in_frame, 0.25);
        for(auto elem : res)
        {
            if (elem.obj_id < Recognizer::OBJ_NAMES.size()) {
                qDebug() << Recognizer::OBJ_NAMES.at(elem.obj_id).c_str()
                         << elem.prob
                         << elem.w
                         << "x"
                         << elem.h;
            } else {
                qDebug() << elem.obj_id;
            }

            static std::set<int> needed {0, 2, 3, 5, 7, 9, 11};
            if (needed.find(elem.obj_id) != needed.end()
                    && (elem.w > 30 || elem.h > 30)) {

                int x_center = elem.x + elem.w / 2;
                int y_center = elem.y + elem.h / 2;
                rectangle(in_frame, Point(elem.x, elem.y), Point(elem.x+elem.w, elem.y+elem.h), red_color, 6);
                putText(in_frame,
                        Recognizer::OBJ_NAMES.at(elem.obj_id).c_str(),
                        Point(x_center, y_center),
                        2,
                        1.,
                        black_color);
                putText(in_frame,
                        Recognizer::OBJ_NAMES.at(elem.obj_id).c_str(),
                        Point(x_center+1, y_center+1),
                        2,
                        1.,
                        white_color);
            }
        }

        putText(in_frame,
                "TEST",
                Point(20, 40),
                2,
                2.,
                road_outline_color);

        // 0.2.3.5.9.11

        writer.write(in_frame);

        cv::imshow("Result", in_frame);

        const auto key = cv::waitKey(1);
        if (key > 0) {
            break;
        }

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // qDebug() << "Frame processed in" << elapsed.count() << "milliseconds";
    }

    in_video.release();
    writer.release();
    qDebug() << "Video saved";
    cv::destroyAllWindows();

    //    QQmlApplicationEngine engine;
    //    engine.rootContext()->setContextProperty("currentDateTime", QDateTime::currentDateTime());
    //    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    //    if (engine.rootObjects().isEmpty())
    //        return -1;

    return 0; // app.exec();
}
