//
// Created by ricardo_8990 on 8/19/15.
// Implementation of the Classifier class
// Classification based on the example provided by the caffe library in:
// https://github.com/BVLC/caffe/blob/master/examples/cpp_classification/classification.cpp
//

#include "Classifier.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "SlideWindowMethod.h"
#include "SelectiveMethod.h"

using std::string;
using namespace caffe;
using namespace cv;
using namespace ml;

Classifier::Classifier(const string &model, const string &weights, ISlideMethod::SlideMethodType type)
{
    //Load network
    net.reset(new Net<float>(model, TEST));
    net->CopyTrainedLayersFrom(weights);

    //Get the number of channels and geometry of the model
    Blob<float>* inputLayer = net->input_blobs()[0];
    numberChannels = inputLayer->channels();
    geometry = cv::Size(inputLayer->width(), inputLayer->height());
    mainImage = Mat();
    runs = 0;
    methodType = type;
}

int Classifier::Classify(const cv::Mat &inputImage)
{
    runs++;
    inputImage.copyTo(mainImage);
    //If there is no previous nests, classify the whole image and return
    if(nests.empty())
    {
        inputImage.copyTo(previousImage);
        return classifyImage(inputImage);
    }

    //Update the position of the existing nests
    Rect innerRect = updateExistingNests(inputImage);
    int minX = innerRect.x;
    int minY = innerRect.y;
    int maxX = innerRect.x + innerRect.width;
    int maxY = innerRect.y + innerRect.height;

    //Calculate the new probability of the existing regions
    for(vector<Nest>::iterator it = nests.begin(); it != nests.end(); ++it)
    {
        float prob = predict(it->rect, inputImage);
        it->probability = ((it->probability * (runs - 1)) + prob) / runs;
    }

    //Extract the part of the images to be processed
    //Start taking the top part
    int nestsNumber = (int) nests.size();
    Mat toProcess;
    if(minY > geometry.height)
    {
        toProcess = inputImage(Rect(0, 0, inputImage.cols, minY));
        nestsNumber += classifyImage(toProcess);
    }
    //Bottom
    if(inputImage.rows - maxY > geometry.height)
    {
        toProcess = inputImage(Rect(0, maxY, inputImage.cols, inputImage.rows - maxY - 1));
        nestsNumber += classifyImage(toProcess, 0, maxY);
    }
    //Left
    if(maxY - minY > geometry.height && minX > geometry.width)
    {
        toProcess = inputImage(Rect(0, minY, minX, maxY - minY));
        nestsNumber += classifyImage(toProcess, 0, minY);
    }
    //Right
    if(maxY - minY > geometry.height && inputImage.cols - maxX > geometry.width)
    {
        toProcess = inputImage(Rect(maxX, minY, inputImage.cols - maxX, maxY - minY));
        nestsNumber += classifyImage(toProcess, maxX, minY);
    }
    //Process images
    mainImage.release();
    inputImage.copyTo(previousImage);
    return nestsNumber;
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
    Blob<float>* outputLayer = net->output_blobs()[0];
    const float* begin = outputLayer->cpu_data();
    const float* end = begin + outputLayer->channels();
    vector<float> results(begin, end);

    return results[1];
}

void Classifier::addPrediction(Rect region, float probability, const Mat &inputImage)
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
        Point2f point = getFeatureToTrack(region);
        nests.push_back(Nest(region, probability, point));
    }
}

Point2f Classifier::getFeatureToTrack(const Rect &region) const
{
    //Get feature to track
    Mat imageRegion = mainImage(region);
    Mat gray;
    cvtColor(imageRegion, gray, CV_RGB2GRAY, 0);
    vector<Point2f> points;
    goodFeaturesToTrack(gray, points, 1, 0.3, 7, Mat(), 7);
    Point2f point = Point2f();
    if(!points.empty())
        point = Point2f(points[0].x + region.x, points[0].y + region.y);
    return point;
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

int Classifier::classifyImage(const cv::Mat &input, int xOffset, int yOffset)
{
    //Select either slide window or selective search method
    ISlideMethod* method;
    if(methodType == ISlideMethod::SlideMethodType::SELECTIVE)
        method = new SelectiveMethod();
    else
        method = new SlideWindowMethod(geometry.height);
    //Initialize slide window
    method->initializeSlideWindow(input);

    while(true)
    {
        //Get a proposed region
        Rect region = method->getProposedRegion();

        if(region.height == 0 && region.width == 0)
            break;

        //Classify it
        float result = predict(region, input);

        //Increase the region with the offset
        region.x += xOffset;
        region.y += yOffset;

        //If it contains a nest add it to the vector
        addPrediction(region, result, input);
    }

    //Return the number of nests
    method->clear();
    delete method;
    return (int) nests.size();
}

Rect Classifier::updateExistingNests(const cv::Mat &input)
{
    //Apply tracking using the Lucas-Kanade algorithm for Optical Flow
    vector<Point2f> points[2];
    vector<Nest*> iterators;
    vector<Nest*> toUpdate;
    //Prepare the features to track contained in each region
    for(vector<Nest>::iterator it = nests.begin(); it != nests.end(); ++it)
    {
        if(it->featToTrack.x != 0 && it->featToTrack.y != 0)
        {
            points[0].push_back(it->featToTrack);
            points[1].push_back(it->featToTrack);
            iterators.push_back(it.base());
        }
        else
        {
            toUpdate.push_back(it.base());
        }
    }
    std::vector<uchar> status(points[1].size());
    std::vector<float> err(points[1].size());
    TermCriteria termCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(31,31);
    //Performs the Lucas-Kanade algorithm
    calcOpticalFlowPyrLK(previousImage, input, points[0], points[1], status, err, winSize, 3, termCriteria, 0, 0.001);

    float dx;
    float dy;
    float avDx = 0;
    float avDy = 0;
    int minX = std::numeric_limits<int>::max(), minY = std::numeric_limits<int>::max(), maxX = 0, maxY = 0;
    //Update the position of each region
    for(unsigned long r = 0; r < points[1].size(); ++r)
    {
        if(!status[r])
        {
            toUpdate.push_back(iterators[r]);
            continue;
        }
        dx = points[1][r].x - points[0][r].x;
        dy = points[1][r].y - points[0][r].y;
        iterators[r]->featToTrack = points[1][r];

        Rect rect = getNewRect(iterators[r]->rect, dx, dy, input, minX, minY, maxX, maxY);
        iterators[r]->rect = rect;
        avDx += dx;
        avDy += dy;
    }
    avDx /= points[1].size();
    avDy /= points[1].size();

    for(int r = 0; r < toUpdate.size(); ++r)
    {
        Rect rect = getNewRect(toUpdate[r]->rect, avDx, avDy, input, minX, minY, maxX, maxY);
        toUpdate[r]->rect = rect;

        toUpdate[r]->featToTrack = getFeatureToTrack(toUpdate[r]->rect);
    }

    for(vector<Nest>::iterator it = nests.begin(); it != nests.end();)
    {
        //If the borders and area is less than the minimum required by the ConvNet, delete it
        if(it->rect.width < 50 || it->rect.height < 50 || it->rect.width > 347 || it->rect.height > 429)
        {
            it = nests.erase(it);
        }
        else
            ++it;
    }

    return Rect(minX, minY, maxX - minX, maxY - minY);
}

cv::Rect Classifier::getNewRect(cv::Rect rect, float dx, float dy, const cv::Mat &inputImage, int &minX, int &minY, int &maxX, int &maxY)
{
    int x, y, width, height;
    float newX = rect.x + dx;
    float newY = rect.y + dy;
    //Check boundaries
    if(newX < 0)
        newX = 0;
    if(newX > inputImage.cols)
        newX = inputImage.cols;
    if(newY < 0)
        newY = 0;
    if(newY > inputImage.rows)
        newY = inputImage.rows;
    x = (int) newX;
    y = (int) newY;

    float newW = rect.x + dx + rect.width;
    float newH = rect.y + dy + rect.height;
    if(newW < 0)
        newW = 0;
    if(newW > inputImage.cols)
        newW = inputImage.cols;
    if(newH < 0)
        newH = 0;
    if(newH > inputImage.rows)
        newH = inputImage.rows;

    width = (int) (newW - x);
    height = (int) (newH - y);

    if(x < minX)
        minX = x;
    if(x + width > maxX)
        maxX = x + width;
    if(y < minY)
        minY = y;
    if(y + height > maxY)
        maxY = y + height;

    return Rect(x, y, width, height);
}
