//
// Created by ricardo_8990 on 8/19/15.
// Implementation of the Classifier class
// Classification based on the example provided by the caffe library in:
// https://github.com/BVLC/caffe/blob/master/examples/cpp_classification/classification.cpp
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Classifier.h"
#include "SlideWindowMethod.h"
#include "SelectiveMethod.h"

using std::string;
using namespace caffe;
using namespace cv;
using namespace ml;

Classifier::Classifier(const string &model, const string &weights)
{
    //Load network
    net.reset(new Net<float>(model, TEST));
    net->CopyTrainedLayersFrom(weights);

    //Get the number of channels and geometry of the model
    Blob<float>* inputLayer = net->input_blobs()[0];
    numberChannels = inputLayer->channels();
    geometry = cv::Size(inputLayer->width(), inputLayer->height());
    method = new SelectiveMethod();
}

int Classifier::Classify(const cv::Mat &image)
{
    //Initialize slide window
    method->initializeSlideWindow(image);

    while(true)
    {
        //Get a proposed region
        Rect region = method->getProposedRegion();

        if(region.height == 0 && region.width == 0)
            break;

        //Classify it
        float result = predict(region, image);

        //If it contains a nest add it to the vector
        addPrediction(region, result);
    }

    //Return the number of nests
    return (int) nests.size();
}

float Classifier::predict(Rect region, const Mat &inputImage)
{
    //Get image from region
    Mat imageRegion = inputImage(region);

    //Forward dimension to layers
    Blob<float>* inputLayer = net->input_blobs()[0];
    inputLayer->Reshape(1, numberChannels, geometry.height, geometry.width);
    net->Reshape();

    //Convert openCVMat to caffe input
    vector<Mat> inputChannels;
    wrapInputLayer(&inputChannels, inputLayer);
    processImage(imageRegion, &inputChannels);

    //Perform prediction
    net->ForwardPrefilled();

    //Return the probability obtained
    //Blob<float>* outputLayer = net->output_blobs()[2];
    Blob<float>* outputLayer = net->output_blobs()[0];
    const float* begin = outputLayer->cpu_data();
    const float* end = begin + outputLayer->channels();
    vector<float> results(begin, end);

    return results[1];
}

void Classifier::addPrediction(Rect region, float probability)
{
    //Check list of regions if overlaps with anyone
    bool isNewElement = true;
    vector<vector<Nest>::iterator> toDelete;
    for(vector<Nest>::iterator it = nests.begin(); it != nests.end(); ++it)
    {
        if((it->rect & region).area() <= 0)
            continue;

        //If it intersects, check the probability
        if(probability <= it->probability)
        {
            isNewElement = false;
            break;
        }

        //Delete the current element if the new probability is higher
        toDelete.push_back(it);
    }

    for(vector<vector<Nest>::iterator>::iterator it = toDelete.begin(); it != toDelete.end(); ++it)
    {
        nests.erase(*it.base());
    }

    if(isNewElement)
    {
        nests.push_back(Nest(region, probability));
    }
}

void Classifier::wrapInputLayer(vector<Mat> *pVector, Blob<float> *pBlob)
{
    float* inputData = pBlob->mutable_cpu_data();
    for (int r = 0; r < pBlob->channels(); ++r)
    {
        Mat channel(pBlob->height(), pBlob->width(), CV_32FC1, inputData);
        pVector->push_back(channel);
        inputData += pBlob->height() * pBlob->width();
    }
}

void Classifier::processImage(Mat image, vector<Mat> *inputChannels)
{
    //Resize the image obtained from the slide method.
    cv::Mat resizeImage;
    if(image.size() != geometry)
        cv::resize(image, resizeImage, geometry);
    else
        resizeImage = image;

    //The number of channels is 3
    Mat imageFormat;
    resizeImage.convertTo(imageFormat, CV_32FC3);

    //Split the image channels into different arrays
    cv::split(imageFormat, *inputChannels);
}