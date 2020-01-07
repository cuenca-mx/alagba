//
// Created by ricardo_8990 on 8/19/15.
//

#ifndef TRACKING_ISLIDEMETHOD_H
#define TRACKING_ISLIDEMETHOD_H

#include <opencv2/core/mat.hpp>

class ISlideMethod
{
public:
    virtual void initializeSlideWindow(cv::Mat image) = 0;
    virtual cv::Rect getProposedRegion() = 0;
    enum SlideMethodType { WINDOW, SELECTIVE };
    virtual void clear() = 0;
};

#endif //TRACKING_ISLIDEMETHOD_H
