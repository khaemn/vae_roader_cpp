#ifndef ROADDETECTOR_H
#define ROADDETECTOR_H

#include <QDebug>

#include <string>
#include <chrono>
#include <vector>

#include <fdeep/fdeep.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class RoadDetector
{
public:
    RoadDetector();

    void load_model(const std::string& filename);
    cv::Mat full_mask(const cv::Mat& input);
    cv::Mat small_mask(const cv::Mat& input);

    cv::Mat main_mask_contour(const cv::Mat& mask);

    cv::Mat predict(const cv::Mat& input);

private:
    fdeep::tensor5 as_vaeroader_input(const cv::Mat& input);

    fdeep::tensor5 raw_predict(const fdeep::tensor5& input);

    cv::Mat as_cv_mat(const fdeep::tensor5& input);

    void postprocess_mask(cv::Mat &input);
    void refine_predicted_mask(cv::Mat &input);
    void crop_with_black(cv::Mat& input, size_t border_thickness = 3);



    static const int NN_HEIGHT = 180;
    static const int NN_WIDTH = 320;

private:
    std::unique_ptr<fdeep::model> m_model { nullptr };
    std::vector<fdeep::model> m_models;

    cv::Mat m_erode_kernel;
    cv::Mat m_dilate_kernel;
};

#endif // ROADDETECTOR_H
