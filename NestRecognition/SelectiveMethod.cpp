//
// Created by ricardo_8990 on 8/19/15.
//

#include "SelectiveMethod.h"

using namespace cv;

SelectiveMethod::SelectiveMethod()
{
}

void SelectiveMethod::initializeSlideWindow(cv::Mat image)
{
    method = new ssm::SelectiveSearchMethod(image, 0.8, 200, 200);
}

cv::Rect SelectiveMethod::getProposedRegion()
{
    return method->getProposedRegion();
}

void SelectiveMethod::clear()
{
    method->clear();
    delete method;
}
