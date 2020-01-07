//
// Created by ricardo_8990 on 8/19/15.
//

#ifndef TRACKING_SELECTIVEMETHOD_H
#define TRACKING_SELECTIVEMETHOD_H

#include "ISlideMethod.h"
#include "SelectiveSearchMethod/SelectiveSearchMethod.h"

class SelectiveMethod : public ISlideMethod
{
public:
    SelectiveMethod();
    void initializeSlideWindow(cv::Mat image);
    cv::Rect getProposedRegion();
    void clear();

private:
    ssm::SelectiveSearchMethod *method;
};

#endif //TRACKING_SELECTIVEMETHOD_H
