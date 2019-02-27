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

static const cv::Scalar road_outline_color( 255, 0, 100 );
static const cv::Scalar red_color( 0, 0, 255 );
static const cv::Scalar yellow_color( 0, 255, 255 );
static const cv::Scalar grey_color( 100, 100, 100 );
static const cv::Scalar black_color( 0, 0, 0 );
static const cv::Scalar white_color( 255, 255, 255 );

void drawRecognized(cv::Mat& in_frame,
                    const std::vector<bbox_t>& objects,
                    int horizontal_offset = 0,
                    int vertical_offset = 0,
                    int min_width = 100,
                    int min_height = 30,
                    int alert_size = 200)
{
    using namespace cv;

    for(auto elem : objects)
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
                && (elem.w >= min_width || elem.h >= min_height)) {

            int x_center = horizontal_offset + elem.x + elem.w / 2;
            int y_center = vertical_offset + elem.y + elem.h / 2;

            int obj_size = std::max(elem.w, elem.h);
            if (elem.obj_id == 0) { // person
                obj_size *= 2;
            }
            int obj_scale = obj_size * 3 / min_width;
            int line_thickness = std::max(1, std::min(15, obj_scale));
            rectangle(in_frame, Rect(horizontal_offset + elem.x,
                                     elem.y + vertical_offset,
                                     elem.w,
                                     elem.h),
                      obj_size > alert_size ? red_color : grey_color,
                      obj_size > alert_size ? -1 : line_thickness);
            putText(in_frame,
                    Recognizer::OBJ_NAMES.at(elem.obj_id).c_str(),
                    Point(x_center, y_center),
                    2,
                    0.7,
                    black_color);
            putText(in_frame,
                    Recognizer::OBJ_NAMES.at(elem.obj_id).c_str(),
                    Point(x_center+1, y_center+1),
                    2,
                    0.7,
                    white_color);
        }
    }
}

int main(int argc, char *argv[])
{
    using namespace cv;
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);

    RoadDetector detector;
    detector.load_model("../resources/micro_model_yolike_roader.json");
    detector.set_approx_epsilon(4.);

    Recognizer recognizer;

    std::vector<std::string> filepaths {
        {"../resources/video/kiev/kiev3.mp4"},
        {"../resources/video/kiev/kiev5.mp4"},
        {"../resources/video/kiev/kiev4.mp4"},
        {"../resources/video/kiev/kiev2.mp4"},
        {"../resources/video/kiev/kiev1.mp4"},
        {"../resources/video/road1.mp4"},
        {"../resources/video/road2.mp4"},
        {"../resources/video/road3.mp4"},
        {"../resources/video/road4.mp4"},
        {"../resources/video/road5.mp4"},
        {"../resources/video/road6.mp4"},
        {"../resources/video/road7.mp4"},
        //{"../resources/video/test/test-road-1.mp4"},
    };

    Size out_size(1280, 720);
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
    constexpr int starting_frame = 300;//3500;//4100;
    constexpr int total_frames = 300;

    for (const auto& filepath: filepaths) {
        int frame_counter = 0;
        int curr_frame = 0;
        cv::VideoCapture in_video = cv::VideoCapture(filepath);

        while (in_video.read(in_frame) && in_video.isOpened()) {
            if (frame_counter++ > frame_divider) {
                frame_counter = 0;
            } else {
                continue;
            }
            curr_frame++;
            if (curr_frame < starting_frame) {
                if (curr_frame % 100 == 0) {
                    qDebug() << "Skipped" << curr_frame << "of" << starting_frame << "frames...";
                }
                continue;
            }

            auto start = std::chrono::system_clock::now();

            resize(in_frame, in_frame, out_size);
            auto road_shape = detector.approx_road_shape(in_frame);

            Mat shape = in_frame.clone();
            if (!road_shape.empty()) {
                std::vector<std::vector<Point>> shapes { road_shape };
                drawContours( shape, shapes, 0, road_outline_color, -1);
                static constexpr float alpha = 0.3;
                addWeighted(shape, alpha, in_frame, 1 - alpha, 0, in_frame);
                drawContours( in_frame, shapes, 0, yellow_color, 1);
            }

            size_t hor_half = (in_frame.cols / 2) - 1;
            size_t third = (in_frame.cols / 3) - 1;
            size_t vertical_offset = ((in_frame.rows - third) / 2) - 1;
            size_t yolo_height = in_frame.rows - 1 - vertical_offset*2;
            Rect left = Rect(0, vertical_offset, third, yolo_height);
            Rect center = Rect(third, vertical_offset, third - 1, yolo_height);
            Rect right = Rect(third * 2, vertical_offset, third - 1, yolo_height);
            Mat left_half = in_frame(left);
            Mat center_part = in_frame(center);
            Mat right_half = in_frame(right);

            // TODO: move semantic
            std::vector<bbox_t> left_res = recognizer.recognize(left_half, 0.35);
            std::vector<bbox_t> center_res = recognizer.recognize(center_part, 0.35);
            std::vector<bbox_t> right_res = recognizer.recognize(right_half, 0.35);

            drawRecognized(in_frame, left_res, 0, vertical_offset, 80, 80, 200);
            drawRecognized(in_frame, center_res, third, vertical_offset, 40, 40, 50);
            drawRecognized(in_frame, right_res, third*2, vertical_offset, 80, 80, 250);

            rectangle(in_frame, left, grey_color, 1);
            rectangle(in_frame, center, grey_color, 1);
            rectangle(in_frame, right, grey_color, 1);

            // DEBUG
            //        Mat raw_mask = detector.small_mask(in_frame);
            //        resize(raw_mask, raw_mask, Size(in_frame.cols, in_frame.rows), INTER_LANCZOS4);
            //        cvtColor(raw_mask, raw_mask, COLOR_GRAY2BGR);
            //        const float alpha = 0.5;
            //        addWeighted(in_frame, alpha, raw_mask, 1 - alpha, 0, in_frame);

            putText(in_frame,
                    "TEST",
                    Point(20, 40),
                    2,
                    1.5,
                    road_outline_color);

            if (writer.isOpened()) {
                writer.write(in_frame);
            }

            cv::imshow("Result", in_frame);

            const auto key = cv::waitKey(1);
            if (key > 0) {
                break;
            }
            if (curr_frame > (starting_frame + total_frames)) {
                break;
            }

            auto end = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            // qDebug() << "Frame processed in" << elapsed.count() << "milliseconds";
        }
        in_video.release();
    }

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
