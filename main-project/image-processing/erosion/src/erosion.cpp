#include <iostream>
#include <opencv2/opencv.hpp>
#include "erosion.hpp"

erosion::erosion()
{
    std::cout << "Erosion Object Constructed" << std::endl;
}

erosion::~erosion()
{
    std::cout << "Erosion Object Destroyed" << std::endl;
}

void erosion::printHello()
{
    std::cout << "Erosion says hello." << std::endl;
}
