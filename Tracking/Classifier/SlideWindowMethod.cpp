//
// Created by ricardo_8990 on 8/19/15.
//

#include "SlideWindowMethod.h"

void SlideWindowMethod::initializeSlideWindow(cv::Mat image)
{
    this->image = image;
}

cv::Rect SlideWindowMethod::getProposedRegion()
{
    //Select columns
    int endColumn = currentColumn + windowSize;
    if(endColumn >= image.cols)
    {
        currentColumn = 0;
        currentRow++;
    }

    //Select rows
    int endRow = currentRow + windowSize;
    if(endRow >= image.rows)
    {
        return cv::Rect();
    }

    //Return region of image
    cv::Rect rect(cv::Point_<int>(currentColumn, currentRow), cv::Point_<int>(endColumn, endRow));
    currentColumn++;
    return rect;
}

SlideWindowMethod::SlideWindowMethod(int windowSize)
{
    this->windowSize = windowSize;
    currentRow = 0;
    currentColumn = 0;
}

void SlideWindowMethod::clear()
{
}
