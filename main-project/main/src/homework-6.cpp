#include "homework-6.hpp"

void showImages(cv::Mat input, cv::Mat output)
{
    using namespace cv;
    imshow("Input", input);

    imshow("Output", output);

    waitKey(0);

    destroyAllWindows();
}