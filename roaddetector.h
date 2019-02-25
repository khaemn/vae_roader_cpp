#ifndef ROADDETECTOR_H
#define ROADDETECTOR_H

#include <string>
#include <chrono>
#include <vector>

// Frugally deep is required to be installed
// https://github.com/Dobiasd/frugally-deep/blob/master/INSTALL.md
#include <fdeep/fdeep.hpp>

// OpenCV is required to be installed
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class RoadDetector
{
public:
    RoadDetector();

    /// @param[in] filename Keras model, converted to json file using frugally-deep
    ///            (https://github.com/Dobiasd/frugally-deep)
    void load_model(const std::string& filename);

    /// @param[in] input OpenCV mat with original frame from car camera
    /// @returns OpenCV mat with road mask (crop) image, white is the road, black is the rest.
    ///          output resolution is equal to native autoencoder resolution.
    ///          OpenCV postprocessing (threshold/erode/dilate) is performed.
    cv::Mat small_mask(const cv::Mat& input);

    /// @param[in] input OpenCV mat with original frame from car camera
    /// @returns OpenCV mat with road mask (crop) image, white is the road, black is the rest.
    ///          output resolution is equal to input resolution.
    ///          OpenCV postprocessing (resize/threshold/erode/dilate) is performed.
    cv::Mat full_mask(const cv::Mat& input);

    /// @param[in] input OpenCV mat with original frame from car camera
    /// @returns OpenCV point vector with the main (largest by area) contour,
    ///          found in the recognized road mask.
    std::vector<cv::Point> road_shape(const cv::Mat& input);

    /// @param[in] input OpenCV mat with original frame from car camera
    /// @returns OpenCV point vector with the main (largest by area) contour,
    ///          found in the recognized road mask and approximated by polyline
    ///          to reduce points count
    std::vector<cv::Point> approx_road_shape(const cv::Mat& input);

    /// Model input resolution is defined by Keras model.
    static const int AUTOENCODER_HEIGHT = 180;
    static const int AUTOENCODER_WIDTH = 320;

    void set_refine_threshold(int refine_threshold);
    void set_approx_epsilon(float approx_epsilon);

private:
    /// @param[in] input OpenCV mat with original frame from car camera
    /// @returns OpenCV mat with road mask (crop) image, white is the road, black is the rest.
    ///          output resolution is equal to native autoencoder resolution.
    cv::Mat predict(const cv::Mat& input);

    /// @param[in] input OpenCV mat with road mask (white - road, black - else)
    /// @returns OpenCV point vector with the main (largest by area) contour,
    ///          found in the recognized road mask.
    std::vector<cv::Point> main_mask_contour(const cv::Mat &mask);

    fdeep::tensor5 as_vaeroader_input(const cv::Mat& input);

    fdeep::tensor5 raw_predict(const fdeep::tensor5& input);

    cv::Mat as_cv_mat(const fdeep::tensor5& input);

    void postprocess_mask(cv::Mat &input);
    void refine_mask(cv::Mat &input);
    void crop_with_black(cv::Mat& input, size_t border_thickness = 1);

    std::vector<cv::Point> approx(const std::vector<cv::Point>& shape);

    /// Resizes given road contour form native
    /// autoencoder resolution to given size
    std::vector<cv::Point> resize_contour(const std::vector<cv::Point>& input, int width, int height);

private:
    std::unique_ptr<fdeep::model> m_model { nullptr };
    std::vector<fdeep::model> m_models;

    cv::Mat m_erode_kernel;
    cv::Mat m_dilate_kernel;

    /// OpenCV RGB threshold for primary mask refinement;
    int m_refine_threshold = 50;

    /// Epsilon (pixel distance) for approximation algorithm
    float m_approx_epsilon = 5.;

    const int m_max_rgb = 255;
};

#endif // ROADDETECTOR_H
