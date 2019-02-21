#include "roaddetector.h"

RoadDetector::RoadDetector()
{
    using namespace cv;

    constexpr int erosion_size = 7;
    m_erode_kernel = getStructuringElement( MORPH_ELLIPSE,
                                            Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                            Point( erosion_size, erosion_size ) );

    constexpr int dilation_size = 7;
    m_dilate_kernel = getStructuringElement( MORPH_ELLIPSE,
                                             Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                             Point( dilation_size, dilation_size ) );
}

void RoadDetector::load_model(const std::string &filename)
{
    m_models.emplace_back(fdeep::load_model(filename));
}

fdeep::tensor5 RoadDetector::as_vaeroader_input(const cv::Mat &input)
{
    cv::Mat greyed;
    cv::cvtColor(input, greyed, CV_RGB2GRAY);
    cv::resize(greyed, greyed, cv::Size(NN_WIDTH, NN_HEIGHT));

    return fdeep::tensor5_from_bytes(greyed.ptr(),
                                     greyed.rows, greyed.cols, greyed.channels());;
}

fdeep::tensor5 RoadDetector::raw_predict(const fdeep::tensor5 &input)
{
    // SEGFAULT PLACE if the model is not loaded:
    return m_models[0].predict({input})[0]; // result is a vector of tensors!;
}

cv::Mat RoadDetector::as_cv_mat(const fdeep::tensor5 &input)
{
    cv::Mat res_image(cv::Size(input.shape().width_, input.shape().height_), CV_8UC1);
    fdeep::tensor5_into_bytes(input,
                              res_image.data,
                              res_image.rows * res_image.cols * res_image.channels());
    return std::move(res_image);
}

void RoadDetector::postprocess_mask(cv::Mat &input)
{
    cv::threshold(input, input, 100, 255, cv::THRESH_BINARY);
}

void RoadDetector::refine_predicted_mask(cv::Mat &input)
{
    cv::dilate(input, input, m_dilate_kernel);
    cv::erode(input, input, m_erode_kernel);
    cv::dilate(input, input, m_dilate_kernel);
    cv::erode(input, input, m_erode_kernel);
}

void RoadDetector::crop_with_black(cv::Mat &input, size_t border_thickness)
{
    // Draws a black rectangle around given mat, thus cropping it;
    assert(input.channels() == 1);

    using namespace cv;

    Point tl (border_thickness, border_thickness);
    Point br (input.cols - border_thickness, input.rows - border_thickness);
    Scalar black (0, 0, 0);
    cv::rectangle(input, tl, br, black);
}

cv::Mat RoadDetector::predict(const cv::Mat &input)
{
    return as_cv_mat(raw_predict(as_vaeroader_input(input)));
}

cv::Mat RoadDetector::full_mask(const cv::Mat &input)
{
    const int orig_width = input.cols;
    const int orig_height = input.rows;

    cv::Mat mask = small_mask(input);

    cv::resize(mask, mask, cv::Size(orig_width, orig_height), 0,0, cv::INTER_LANCZOS4);

    postprocess_mask(mask);

    return mask;
}

cv::Mat RoadDetector::small_mask(const cv::Mat &input)
{
    auto _small_mask = as_cv_mat(raw_predict(as_vaeroader_input(input)));

    refine_predicted_mask(_small_mask);

    return _small_mask;
}

cv::Mat RoadDetector::main_mask_contour(const cv::Mat &mask)
{
    using namespace cv;
    using namespace std;

    static constexpr size_t thresh = 100;
    static constexpr size_t max_thresh = 255;
    static constexpr size_t cropping_offset = 2; // pixels

    Mat canny_output = mask;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Crop plot to achieve visible border for a contour
    crop_with_black(canny_output);
    /// Detect edges using canny
    Canny( mask, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        Scalar color = Scalar( 255, 255, 255 );
        drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
    }

    return drawing;
}
