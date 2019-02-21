#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QDateTime>
#include <QDebug>

#include <iostream>
#include <string>
#include <chrono>

#include <fdeep/fdeep.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "roaddetector.h"

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);

    RoadDetector detector;
    detector.load_model("../resources/mini_model_yolike_roader.json");

    cv::VideoCapture in_video = cv::VideoCapture("../resources/test.mp4");
    cv::Mat in_frame;

    constexpr int frame_divider = 10;
    int frame_counter = 0;

    while (in_video.read(in_frame)) {
        if (frame_counter++ > frame_divider) {
            frame_counter = 0;
        } else {
            continue;
        }

        cv::imshow("Video", in_frame);

        auto start = std::chrono::system_clock::now();

        const cv::Mat mask = detector.small_mask(in_frame);

        // const cv::Mat drawn = detector.main_mask_contour(mask);
        cv::imshow("Result", mask);

        const auto key = cv::waitKey(1);
        if (key > 0) {
            break;
        }

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "Frame processed in" << elapsed.count() << "milliseconds";
    }

    in_video.release();
    cv::destroyAllWindows();

//    QQmlApplicationEngine engine;
//    engine.rootContext()->setContextProperty("currentDateTime", QDateTime::currentDateTime());
//    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
//    if (engine.rootObjects().isEmpty())
//        return -1;

    return app.exec();
}
