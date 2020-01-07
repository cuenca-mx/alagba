//
// Created by ricardo_8990 on 8/19/15.
//

#ifndef TRACKING_CLASSIFIER_H
#define TRACKING_CLASSIFIER_H

#include <bits/stringfwd.h>
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
    cv::Point2f featToTrack;

    inline Nest(cv::Rect rect, float probability, cv::Point2f featToTrack)
    {
        this->rect = rect;
        this->probability = probability;
        this->featToTrack = featToTrack;
    }
};

/*
 * Classifier class, it creates the structure for the ConvNet architecture and detect nests from an image using the
 * Classify function which receives each frame.
 */
class Classifier
{
public:
    Classifier(const std::string& model, const std::string& weights, ISlideMethod::SlideMethodType type);
    int Classify(const cv::Mat&inputImage);
    std::vector<Nest> nests;

private:
    std::shared_ptr<caffe::Net<float>> net;
    int numberChannels;
    cv::Size geometry;
    cv::Mat previousImage;
    cv::Mat mainImage;
    int runs;
    ISlideMethod::SlideMethodType methodType;

private:
    float predict(cv::Rect region, const cv::Mat &inputImage);
    void addPrediction(cv::Rect region, float probability, const cv::Mat &image);
    void wrapInputLayer(std::vector<cv::Mat> *pVector, caffe::Blob<float> *pBlob);
    void processImage(cv::Mat image, std::vector<cv::Mat> *inputChannels);
    int classifyImage(const cv::Mat &input, int xOffset = 0, int yOffset = 0);
    cv::Rect2i updateExistingNests(const cv::Mat &input);
    cv::Point2f getFeatureToTrack(const cv::Rect &region) const;
    cv::Rect getNewRect(cv::Rect rect, float dx, float dy, const cv::Mat &inputImage, int &minX, int &minY, int &maxX, int &maxY);
};

#endif //TRACKING_CLASSIFIER_H
