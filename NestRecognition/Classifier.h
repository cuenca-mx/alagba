//
// Created by ricardo_8990 on 7/28/15.
//

#ifndef NESTRECOGNITION_CLASSIFIER_H
#define NESTRECOGNITION_CLASSIFIER_H

#include <bits/stringfwd.h>
#include <bits/stl_bvector.h>
#include <bits/stl_pair.h>
#include <opencv2/core/mat.hpp>
#include <caffe/caffe.hpp>
#include <opencv2/ml.hpp>
#include "ISlideMethod.h"

/*
 * Nest class, contain all regions obtained from the Object recognition task with a probability to be a nests.
 */
struct Nest
{
    cv::Rect rect;
    float probability;

    inline Nest(cv::Rect rect, float probability)
    {
        this->rect = rect;
        this->probability = probability;
    }
};

/*
 * Classifier class, it creates the structure for the ConvNet architecture and detect nests from an image using the
 * Classify function which receives each frame.
 */
class Classifier
{
public:
    Classifier(const std::string& model, const std::string& weights);
    int Classify(const cv::Mat& image);

    std::vector<Nest> nests;
private:
    float predict(cv::Rect region, const cv::Mat &inputImage);
    void addPrediction(cv::Rect region, float probability);

private:
    std::shared_ptr<caffe::Net<float>> net;
    int numberChannels;
    cv::Size geometry;
    ISlideMethod *method;

    void wrapInputLayer(std::vector<cv::Mat> *pVector, caffe::Blob<float> *pBlob);

    void processImage(cv::Mat image, std::vector<cv::Mat> *inputChannels);
};


#endif //NESTRECOGNITION_CLASSIFIER_H
