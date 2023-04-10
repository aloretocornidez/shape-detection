#include "dilation.hpp"

#include <opencv2/opencv.hpp>

dilation::dilation()
{
    std::cout << "Dilation Object Constructed" << std::endl;
}

dilation::~dilation()
{
    std::cout << "Dilation Object Destroyed" << std::endl;

}

void dilation::printHello()
{
    std::cout << "Dilation says hello." << std::endl;
}
