//
// Created by ricardo_8990 on 8/19/15.
//

#ifndef TRACKING_SLIDEWINDOWMETHOD_H
#define TRACKING_SLIDEWINDOWMETHOD_H

#include "ISlideMethod.h"

class SlideWindowMethod : public ISlideMethod
{
public:
    SlideWindowMethod(int windowSize);
    void initializeSlideWindow(cv::Mat image);
    cv::Rect getProposedRegion();
    void clear();

private:
    int currentRow;
    int currentColumn;
    int windowSize;
    cv::Mat image;
};

#endif //TRACKING_SLIDEWINDOWMETHOD_H
