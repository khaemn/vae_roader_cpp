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


fdeep::tensor5 to_vae_roader_input(const cv::Mat& input)
{
    cv::Mat greyed;
    cv::cvtColor(input, greyed, CV_RGB2GRAY);
    cv::resize(greyed, greyed, cv::Size(320, 180));
//    const fdeep::tensor5 tensor =
//            fdeep::tensor5_from_bytes(greyed.ptr(),
//                                      greyed.rows, greyed.cols, greyed.channels());
    return fdeep::tensor5_from_bytes(greyed.ptr(),
                                     greyed.rows, greyed.cols, greyed.channels());;
}

fdeep::tensor5 get_mnist_image_as_tensor(const std::string& filename)
{
    const cv::Mat image1 = cv::imread(filename);
    cv::Mat greyed;
    cv::cvtColor(image1, greyed, CV_RGB2GRAY);

    const fdeep::tensor5 tensor =
            fdeep::tensor5_from_bytes(greyed.ptr(),
                                      greyed.rows, greyed.cols, greyed.channels());

    // choose the correct pixel type for cv::Mat (gray or RGB/BGR)
    assert(tensor.shape().depth_ == 1 || tensor.shape().depth_ == 3);
    const int mat_type = tensor.shape().depth_ == 1 ? CV_8UC1 : CV_8UC3;

    // convert fdeep::tensor5 to cv::Mat (tensor to image2)
    const cv::Mat image2(
                cv::Size(tensor.shape().width_, tensor.shape().height_), mat_type);
    fdeep::tensor5_into_bytes(tensor,
                              image2.data, image2.rows * image2.cols * image2.channels());
    cv::imshow("Input", greyed);

    return tensor;
}


int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);

    // auto input = get_mnist_image_as_tensor("1.jpg");

    const auto model = fdeep::load_model("cl_model_yolike_roader.json");

    cv::VideoCapture in_video = cv::VideoCapture("test-road-1.mp4");
    cv::Mat in_frame;

    while (in_video.read(in_frame))
    {
        cv::imshow("Video", in_frame);

        auto pred_input = to_vae_roader_input(in_frame);

        auto start = std::chrono::system_clock::now();

        const auto result = model.predict({pred_input})[0]; // result is a vector of tensors!

        const cv::Mat res_image(cv::Size(result.shape().width_,
                                         result.shape().height_),
                                result.shape().depth_ == 1 ? CV_8UC1 : CV_8UC3);

        fdeep::tensor5_into_bytes(result,
                                  res_image.data, res_image.rows * res_image.cols * res_image.channels());

        cv::imshow("Result", res_image);

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        qDebug() << "Frame processed in" << elapsed.count() << "milliseconds";

        const auto key = cv::waitKey(5);
        if (key > 0) {
            break;
        }
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
